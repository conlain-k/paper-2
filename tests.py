import os

from helpers import *


from constlaw import *
from tensor_ops import *
from plot_cube import *
from euler_ang import *
from greens_op import *
from loaders import *
from config import Config
from solvers import make_localizer, LocalizerFFT, IFNOLocalizer

import torch
import time
from h5py import File

from main import load_data

import numpy as np
import matplotlib.ticker as tck

torch.set_flush_denormal(True)

TIMINGS = {
    "FNO": 0.388 / 16.0,
    "Big FNO": 1.101 / 16.0,
    "IFNO": 3.568 / 16.0,
    "FNO-DEQ": 6.549 / 16.0,
    "Mod. F-D": 6.408 / 16.0,
    "TherINO": 7.460 / 16.0,
    "FFT": 1.7 / 16.0,  # highly estimated
}

PLOT_IND_BAD = 1744

BS = 32
N = 31
k = 2 * PI / N

E_VALS = [120.0, 100 * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = torch.as_tensor([0.001, 0, 0, 0, 0, 0]).reshape(1, 6, 1, 1, 1)

C11, C12, C44 = 2000, 1000, 2000

TOL = 1e-6

C_ROT_MOOSE = torch.tensor(
    [
        [
            [179.0177, 62.2070, 58.7752, 2.7766, 2.7475, 2.6219],
            [62.2070, 174.1463, 63.6466, -4.1537, -9.8078, -3.1191],
            [58.7752, 63.6466, 177.5781, 1.3771, 7.0604, 0.4972],
            [2.7766, -4.1537, 1.3771, 107.2932, 0.7031, -13.8704],
            [2.7474, -9.8078, 7.0604, 0.7031, 97.5505, 3.9267],
            [2.6219, -3.1191, 0.4972, -13.8704, 3.9267, 104.4140],
        ]
    ]
)

CHECK_FF = "checkpoints/model_ff_18.9M_s32_fixed32_best.ckpt"
CONF_FF = "configs/ff.json"

CHECK_FNODEQ = "checkpoints/model_fno_deq_18.9M_s32_fixed32_best.ckpt"
CONF_FNODEQ = "configs/fno_deq.json"

CHECK_THERINO = "checkpoints/model_therino_18.9M_s32_fixed32_best.ckpt"
CONF_THERINO = "configs/therino.json"

CHECK_THERNOTHER = "checkpoints/model_therino_notherm_18.9M_s32_fixed32_best.ckpt"
CONF_THERNOTHER = "configs/therino_notherm.json"

CHECK_THERPRE = "checkpoints/model_therino_pre_18.9M_s32_fixed32_best.ckpt"
CONF_THERPRE = "configs/therino_pre.json"

CHECK_THERPOST = "checkpoints/model_therino_post_18.9M_s32_fixed32_best.ckpt"
CONF_THERPOST = "configs/therino_post.json"

CHECK_THERHYBRID = "checkpoints/model_therino_hybrid_18.9M_s32_fixed32_best.ckpt"
CONF_THERHYBRID = "configs/therino_hybrid.json"

CHECK_IFNO = "checkpoints/model_ifno_18.9M_s32_fixed32_best.ckpt"
CONF_IFNO = "configs/ifno.json"

CONF_FFT = "configs/fft.json"
CHECK_FFT = "checkpoints/fft.ckpt"


CHECK_TEST = CHECK_THERINO
CONF_TEST = CONF_THERINO


def batched_vector_grad(a):
    # assumes f is [b x d x i x j x k]
    # b: batch index
    # d: channel of output (e.g. displacement component)
    # i, j, k: spatial dimensions

    # first take fourier transform of signal (for each batch and channel)
    a_FT = torch.fft.fftn(a, dim=(-3, -2, -1))
    # assume all spatial dims are the same
    n = a.shape[-1]
    filt = torch.fft.fftfreq(n) * n * k
    # x-filter affects x-direction, etc.
    filt_x = filt.reshape(1, 1, -1, 1, 1)
    filt_y = filt.reshape(1, 1, 1, -1, 1)
    filt_z = filt.reshape(1, 1, 1, 1, -1)

    da_dx_FT = a_FT * filt_x * 1j
    da_dy_FT = a_FT * filt_y * 1j
    da_dz_FT = a_FT * filt_z * 1j
    # add a dimension for gradient BEFORE channel
    grad_a_FT = torch.stack([da_dx_FT, da_dy_FT, da_dz_FT], axis=1)

    grad_a = torch.fft.ifftn(grad_a_FT, dim=(-3, -2, -1))
    grad_a = grad_a.real
    return grad_a[:, 0], grad_a[:, 1], grad_a[:, 2]


def test_mat_vec_op():
    vec = torch.randn(BS, 6, N, N, N)
    mat = mandel_to_mat_3x3(vec)
    vec2 = mat_3x3_to_mandel(mat)
    mat2 = mandel_to_mat_3x3(vec)

    # make sure we didn't actually change any values
    torch.testing.assert_close(vec, vec2)
    torch.testing.assert_close(mat, mat2)


def test_constlaw_2phase():
    m_base = "paper2_16"
    r_base = "paper2_16_u1_responses"

    E_VALS = [120.0, 100 * 120.0]
    NU_VALS = [0.3, 0.3]

    print(E_VALS, NU_VALS)

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.TRAIN])
    dataset.assignConstlaw(constlaw)

    inds = [0, 5, 120, 690, 1111]
    C_field, _, strain, stress = dataset[inds]

    # C_field = constlaw.compute_C_field(micro)
    check_constlaw(constlaw, C_field, strain, stress)


def test_constlaw_cubic():
    f = File("01_CubicSingleEquiaxedOut.dream3d")

    euler_ang = f["DataContainers"]["SyntheticVolumeDataContainer"]["CellData"][
        "EulerAngles"
    ][:]

    euler_ang = torch.from_numpy(euler_ang)

    # print(euler_ang[:2, :2, :2])

    # stiffness_tens_file = "~/scratch/outputs/poly_euler_ang_0002.csv"

    # stiffnesses = np.genfromtxt(stiffness_tens_file, delimiter=",")

    # stiff_compare = stiffnesses[]

    constlaw = StrainToStress_crystal(C11, C12, C44)

    C_field = constlaw.compute_C_field(euler_ang[None])

    print(C_field[0, :, :, 11, 0, 0])

    # now compare to true solution
    true_resp_file = "/storage/home/hcoda1/3/ckelly84/scratch/poly_stress_strain.h5"

    with h5py.File(true_resp_file, "r") as f:
        strain_true = to_mandel(torch.as_tensor(f["strain"][:]).float(), fac=SQRT2)

        stress_true = to_mandel(torch.as_tensor(f["stress"][:]).float(), fac=SQRT2)

        check_constlaw(constlaw, C_field.float(), strain_true, stress_true)


def test_fft_deriv():
    KMAX = 4

    def f(x, omega=1):
        # Assume spatial dim comes first
        return torch.sin(omega * k * x[0]) + torch.cos(2 * omega * k * x[1])

    def grad_f(x, omega=1):
        f_x = omega * k * torch.cos(omega * k * x[0])
        f_y = -2 * omega * k * torch.sin(2 * omega * k * x[1])
        f_z = 0 * x[2]

        return f_x, f_y, f_z

    # f = lambda X, k: torch.sin(k * X[0]) + torch.cos(k * X[1]) + 0 * X[2]

    # def grad_f(X, k):
    #     g_x = k * torch.cos(k * X[0])
    #     g_y = -k * torch.sin(k * X[1])
    #     g_z = 0 * X[2]

    #     return g_x, g_y, g_z

    # get x locs
    x = torch.arange(0, N)
    # get 3d grid
    X = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=0)

    # print(X.shape)

    fX = torch.zeros(KMAX, 3, N, N, N)
    gfx = torch.zeros(KMAX, 3, N, N, N)
    gfy = torch.zeros(KMAX, 3, N, N, N)
    gfz = torch.zeros(KMAX, 3, N, N, N)

    for i in range(KMAX):
        omega = i
        fX[i] = f(X, omega)
        gfx[i], gfy[i], gfz[i] = grad_f(X, omega)

    dx, dy, dz = batched_vector_FFT_grad(fX, disc=True)

    print("dx", dx[1, 1, 2, 3])
    print("gfx", gfx[1, 1, 2, 3])

    import matplotlib.pyplot as plt

    from train import plot_pred

    # plot_pred(-1, torch.zeros(N, N), gfx[3, 0, :, :, 0], dx[3, 0, :, :, 0], "dx")

    torch.testing.assert_close(gfx, dx)
    torch.testing.assert_close(gfy, dy)
    torch.testing.assert_close(gfz, dz)


def prof_C_op():
    # profile stress computation using C and m

    micros = torch.randn(128, 2, 31, 31, 31)
    C_op = StrainToStress_2phase([1, 1000], [0.3, 0.3])
    strains = torch.randn(128, 6, 31, 31, 31)

    def time_op(f):
        torch.cuda.synchronize()
        t_start = time.time()
        f()
        torch.cuda.synchronize()
        return time.time() - t_start

    def op_simul():
        for i in range(64):
            stress = torch.einsum(
                "bhxyz, hrc, bcxyz->brxyz", micros, C_op.stiffness_mats, strains
            )

    def op_premul():
        C_big = torch.einsum("bhxyz, hrc -> brcxyz", micros, C_op.stiffness_mats)
        for i in range(64):
            stress = torch.einsum("brcxyz, bcxyz->brxyz", C_big, strains)

    op_simul()
    op_premul()
    op_simul()
    op_premul()

    time_simul = time_op(op_simul)
    time_premul = time_op(op_premul)

    print(f"Simultaneous time is {time_simul:.3f}, premul time is {time_premul:.3f}")


def test_euler_ang():

    f = File("01_CubicSingleEquiaxedOut.dream3d")

    euler_ang = f["DataContainers"]["SyntheticVolumeDataContainer"]["CellData"][
        "EulerAngles"
    ][:]

    euler_ang = torch.from_numpy(euler_ang)

    print(euler_ang.shape)

    constlaw = StrainToStress_crystal(C11, C12, C44)

    C_field = constlaw.compute_C_field(euler_ang[None])

    # print(C_field.shape)

    # plot c11
    plot_cube(C_field[0, 0, 0], "d3d_test.png")


def test_mandel():
    C_orig = cubic_mandel66(1, 2, 3)
    C_3333 = C_mandel_to_mat_3x3x3x3(C_orig)
    C_66 = C_3x3x3x3_to_mandel(C_3333)
    C_3333_again = C_mandel_to_mat_3x3x3x3(C_66)

    assert (C_orig - C_66).abs().max() <= TOL
    assert (C_3333 - C_3333_again).abs().max() <= TOL


def test_rotat():
    # make sure that rotation gives expected result
    def check_rot(C_in, C_out_true, euler_ang):
        # print(euler_ang.shape, C_in.shape)
        C_out = batched_rotate(euler_ang, C_in)

        # print(C_out_true)
        # print(C_out)

        err = (C_out_true - C_out).abs() / C_out_true.abs().max()
        if err.max() > TOL:
            print(C_3x3x3x3_to_mandel(C_out_true))
            print(C_3x3x3x3_to_mandel(C_out))
            print(C_3x3x3x3_to_mandel(err))

        assert err.max() <= TOL

    # rotating identity = no-op
    C_ident = C_mandel_to_mat_3x3x3x3(cubic_mandel66(1, 0, 1))
    euler_ang = torch.randn(5, 3)

    # print("Ci", C_ident.shape)

    # make sure that arbitrary rotations don't affect identity
    # check_rot(C_ident, C_ident, euler_ang)

    # add off-diag entries
    C_1 = C_mandel_to_mat_3x3x3x3(cubic_mandel66(111, 22, 44))

    euler_ang = torch.zeros(1, 3)
    euler_ang[:, 0] = 0
    euler_ang[:, 1] = 0
    # 90 degree rotation = no change
    euler_ang[:, 2] = torch.pi / 2.0

    check_rot(C_1, C_1, euler_ang)

    # ditto 180 and 270, for each dir
    euler_ang[:, 2] = torch.pi
    check_rot(C_1, C_1, euler_ang)

    euler_ang[:, 2] = 3 * torch.pi / 2
    check_rot(C_1, C_1, euler_ang)

    euler_ang[:, 0] = torch.pi / 2
    euler_ang[:, 2] = 0
    check_rot(C_1, C_1, euler_ang)

    euler_ang[:, 0] = 0
    euler_ang[:, 1] = torch.pi / 2
    check_rot(C_1, C_1, euler_ang)

    # pi/4 twice = nothing
    euler_ang[:, 0] = torch.pi / 4
    euler_ang[:, 1] = 0
    euler_ang[:, 2] = torch.pi / 4
    check_rot(C_1, C_1, euler_ang)

    # second undoes first
    euler_ang[:, 0] = -torch.pi / 3
    euler_ang[:, 1] = 0
    euler_ang[:, 2] = torch.pi / 3
    check_rot(C_1, C_1, euler_ang)


def test_stiff_ref():
    C_unrot = C_mandel_to_mat_3x3x3x3(cubic_mandel66(160, 70, 60))
    euler_ang = torch.tensor([[PI / 4, PI / 3, -PI / 3]])

    C_rot3333 = batched_rotate(euler_ang, C_unrot)
    C_rot66 = C_3x3x3x3_to_mandel(C_rot3333)

    # print(C_rot66)

    err = (C_rot66 - C_ROT_MOOSE).abs() / C_ROT_MOOSE.abs().max()

    # make sure we match tensor that MOOSE gave us
    assert err.max() <= TOL


def test_euler_pred(conf_curr=CONF_THERINO, check_curr=CHECK_THERINO):

    Z = 2 * C44 / (C11 - C12)

    print(f"Zener ratio is {Z:.2f}")

    constlaw = StrainToStress_crystal(C11, C12, C44)
    # pull in saved weights
    if check_curr is not None:
        model = load_checkpoint(check_curr, None, strict=True)
        model.config.num_voxels = 64
        model.overrideConstlaw(constlaw)
    else:
        conf_args = load_conf_override(conf_curr)
        config = Config(**conf_args)
        config.num_voxels = 64

        model = make_localizer(config, constlaw)
    # no need to return residual
    model.config.add_resid_loss = False
    fac = 1

    # rebuild model scalings using new constlaw

    print(model.stiffness_scaling)
    model.compute_scalings(E_BAR)
    print(model.stiffness_scaling)

    model.greens_op = GreensOp(model.constlaw, 64 * fac)

    # make sure we're ready to eval
    # model = model.cuda()
    model.eval()

    import time

    print("running inference!")

    from main import dataset_info

    loader = load_data(dataset_info["poly64"], DataMode.TEST, constlaw)

    C_field, bc_vals, strain_true, stress_true = loader.dataset[0:1]

    homog_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()

    print("C mean unnormalized", C_field[0].mean((-3, -2, -1)))
    print("C ref", constlaw.C_ref)
    print("C mean normalized", model.scale_stiffness(C_field[0].mean((-3, -2, -1))))

    model = model.cuda()
    C_field = C_field.cuda()
    bc_vals = bc_vals.cuda()

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        # evaluate on batch of euler angles
        strain_pred = model.forward(C_field, bc_vals)

    torch.cuda.synchronize()
    dt = time.time() - start

    # print(strain_pred.shape)

    print("elapsed time", dt)

    C_field = C_field.cpu()
    strain_pred = strain_pred.cpu()
    model = model.cpu()

    stress_pred = model.constlaw(C_field, strain_pred)

    homog_pred = est_homog(strain_pred, stress_pred, (0, 0)).squeeze()

    # now compare to true solution
    print(
        f"\nHomog true: {homog_true:4f} pred: {homog_pred:4f} rel_err: {(homog_true - homog_pred)/homog_true:4f}"
    )

    L2_strain_err = model.scale_strain(strain_pred - strain_true) ** 2
    L2_strain_err_val = L2_strain_err.sum(1).mean((-3, -2, -1)).mean()

    equiv_strain_true = equivalent(strain_true).detach().cpu().squeeze()
    equiv_strain_pred = equivalent(strain_pred).detach().cpu().squeeze()

    equiv_stress_true = equivalent(stress_true).detach().cpu().squeeze()
    equiv_stress_pred = equivalent(stress_pred).detach().cpu().squeeze()

    plot_cube(
        100 * model.scale_strain(equiv_strain_true - equiv_strain_pred).abs(),
        f"poly_extrap_{model.config.arch_str}_strain_err.png",
        cmap="turbo",
        title="Percent L1 Equivalent Strain Error",
    )

    plot_cube(
        100 * model.scale_stress(equiv_stress_true - equiv_stress_pred).abs(),
        f"poly_extrap_{model.config.arch_str}_stress_err.png",
        cmap="turbo",
        title="Percent L1 Equivalent Stress Error",
    )

    plot_cube(
        equiv_strain_pred[:, :, :],
        f"poly_extrap_{model.config.arch_str}_strain.png",
        cmap="turbo",
        title=f"{model.config.arch_str}",
    )
    plot_cube(
        equiv_stress_pred[:, :, :],
        f"poly_extrap_{model.config.arch_str}_stress.png",
        cmap="turbo",
    )

    plot_cube(
        equiv_strain_true[:, :, :],
        "poly_true_strain.png",
        cmap="turbo",
    )
    plot_cube(
        equiv_stress_true[:, :, :],
        "poly_true_stress.png",
        cmap="turbo",
    )

    ipf_colors = loader.dataset.getData(loader.dataset.mf, "IPFColor")[0:1]

    print(ipf_colors.shape)

    plot_cube(ipf_colors / 256.0, "poly_ipf.png", cmap=None, add_cb=False)

    plot_cube(
        C_field[0, 0, 0, :, :, :].detach().cpu(),
        "d3d_C11.png",
        cmap="gray_r",
        title="$C_{1111}$",
    )

    # # dump predictions to a file
    # f = File("crystal_pred.h5", "w")
    # write_dataset_to_h5(C_field, "C_field", f)
    # write_dataset_to_h5(strain_pred, "strain", f)
    # write_dataset_to_h5(stress_pred, "stress", f)
    # write_dataset_to_h5(euler_ang.unsqueeze(0), "euler_ang", f)


def test_model_save_load():
    conf_args = load_conf_override(CONF_TEST)
    # conf_args["fno_args"]["modes"] = (16, 16)
    # print(conf_args)
    config = Config(**conf_args)
    config.num_voxels = 32

    # no need to return residual
    config.add_resid_loss = False
    # config.fno_args["normalize_inputs"] = False
    # config.fno_args["use_mlp_lifting"] = False

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)
    # make first model
    model = make_localizer(config, constlaw)

    model = load_checkpoint(CHECK_TEST, model, strict=True)

    model.eval()
    model.cuda()

    TEST_PATH = "checkpoints/test.ckpt"

    # save these (randomly-init) params
    save_checkpoint(model, None, None, 0, 0, path_override=TEST_PATH, backup_prev=False)

    # now try loading the re-saved model
    model_2 = make_localizer(config, constlaw)

    model_2 = load_checkpoint(TEST_PATH, model_2, strict=True)
    model_2.eval()
    model_2.cuda()

    bc_vals = E_BAR.cuda()

    # make fake inputs to test model
    fake_inputs = torch.randn(2, 6, 6, 32, 32, 32).cuda()

    print("Comparing model inference results!")
    with torch.inference_mode():
        outputs_orig = model(fake_inputs, bc_vals)
        outputs_new = model_2(fake_inputs, bc_vals)

    assert torch.allclose(outputs_orig, outputs_new)

    # if model.state_dict() == model_2.state_dict():
    #     print("ok")

    for k in model.state_dict().keys():

        if torch.is_tensor(model.state_dict()[k]):
            ok = (model.state_dict()[k] == model_2.state_dict()[k]).all()
        else:
            ok = model.state_dict()[k] == model_2.state_dict()[k]
        if not ok:
            print(f"Key {k} does not match!!")
            os.remove(TEST_PATH)
            exit(1)

    print("Model save load passed!")

    # clean up after ourselves
    os.remove(TEST_PATH)


def test_FFT_iters_crystal():

    f = File("01_CubicSingleEquiaxedOut.dream3d")

    euler_ang = f["DataContainers"]["SyntheticVolumeDataContainer"]["CellData"][
        "EulerAngles"
    ][:]

    euler_ang = torch.from_numpy(euler_ang)

    constlaw = StrainToStress_crystal(C11, C12, C44)

    C_field = constlaw.compute_C_field(euler_ang[None])

    g_solver = LocalizerFFT(Config(num_voxels=62), constlaw)

    import matplotlib as mpl

    mpl.rcParams["figure.facecolor"] = "white"

    # now compare to true solution
    true_resp_file = "/storage/home/hcoda1/3/ckelly84/scratch/poly_stress_strain.h5"

    eps = E_BAR.expand(1, 6, 62, 62, 62)

    with h5py.File(true_resp_file, "r") as f:
        strain_true = torch.as_tensor(f["strain"][:]).reshape(eps.shape).to(eps.device)
        stress_true = torch.as_tensor(f["stress"][:]).reshape(eps.shape).to(eps.device)
        homog_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()

    print(f"True Cstar {homog_true:4f}")

    if torch.cuda.is_available():
        C_field = C_field.cuda()
        eps = eps.cuda()
        eps_bar = E_BAR.cuda()
        constlaw = constlaw.cuda()
        g_solver = g_solver.cuda()

    traj = g_solver._compute_trajectory(C_field, eps_bar, num_iters=10)

    Cstar = [
        est_homog(strain, constlaw(C_field, strain), (0, 0)).squeeze()
        for strain in traj
    ]

    for ind, strain in enumerate(traj):
        plot_cube(
            equivalent(strain),
            savedir=f"fft_strain_{ind}.png",
            cmap="coolwarm",
            vmax=equivalent(strain_true).max(),
        )

    print("Cstar", Cstar)


def test_FFT_iters_2phase():

    m_base = "paper2_smooth"
    r_base = None

    m_base = "paper2_16"
    r_base = "paper2_16_u2_responses"
    UPSAMP_MICRO_FAC = 2

    constlaw = StrainToStress_2phase([120, 120 * 100], [0.3, 0.3])

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(
        **datasets[DataMode.TRAIN], upsamp_micro_fac=UPSAMP_MICRO_FAC
    )
    dataset.assignConstlaw(constlaw)
    C_field, bc_vals, eps_FEA, sigma_FEA = dataset[0:1]

    # resp_f = h5py.File("/storage/home/hcoda1/3/ckelly84/scratch/outputs/spher_inc_cr100.0_bc0/00000.h5")
    # eps_FEA, sigma_FEA = resp_f["strain"][:], resp_f["stress"][:]

    # resp_f.close()

    # m = h5py.File("/storage/home/hcoda1/3/ckelly84/scratch/micros/spher_inc.h5")["micros"][:]

    C_field = torch.as_tensor(C_field)
    eps_FEA = torch.as_tensor(eps_FEA)
    sigma_FEA = torch.as_tensor(sigma_FEA)

    N = C_field.shape[-1]

    # C_field = constlaw.compute_C_field(m)

    C_homog_FEA = sigma_FEA[0, 0].mean() / eps_FEA[0, 0].mean()

    G = GreensOp(constlaw, N)

    eps = torch.zeros(1, 6, N, N, N)
    eps[:, 0] = 0.001

    import matplotlib as mpl

    mpl.rcParams["figure.facecolor"] = "white"

    plot_cube(eps_FEA[0, 0], savedir="2phase_eps_FEA.png", cmap="coolwarm")
    plot_cube(sigma_FEA[0, 0], savedir="2phase_sigma_FEA.png", cmap="coolwarm")

    FEA_resid_equi, FEA_resid_compat = G.compute_residuals(eps_FEA, sigma_FEA)
    FEA_div_sigma_FT = stressdiv(sigma_FEA).mean()
    FEA_equi_err = constlaw.C0_norm(FEA_resid_equi).mean()
    FEA_compat_err = constlaw.C0_norm(FEA_resid_compat).mean()

    print(
        f"FEA Residuals, div sigma = {FEA_div_sigma_FT:4f}, equi err = {FEA_equi_err:4f}, compat err = {FEA_compat_err:4f}"
    )

    sigma_init = constlaw(C_field, eps)

    init_resid_equi, init_resid_compat = G.compute_residuals(eps, sigma_init)
    init_div_sigma_FT = stressdiv(sigma_init).mean()
    init_equi_err = constlaw.C0_norm(init_resid_equi).mean()
    init_compat_err = constlaw.C0_norm(init_resid_compat).mean()

    print(
        f"init Residuals, div sigma = {init_div_sigma_FT:4f}, equi err = {init_equi_err:4f}, compat err = {init_compat_err:4f}"
    )

    # get initial field with correct mean (not in equilib)
    eps_rand = torch.randn(1, 6, N, N, N)
    eps_rand = eps + eps_rand - eps_rand.mean(dim=(-3, -2, -1), keepdim=True)
    sigma_rand = constlaw(C_field, eps_rand)

    rand_resid_equi, rand_resid_compat = G.compute_residuals(eps_rand, sigma_rand)
    rand_div_sigma_FT = stressdiv(sigma_rand).mean()
    rand_equi_err = constlaw.C0_norm(rand_resid_equi).mean()
    rand_compat_err = constlaw.C0_norm(rand_resid_compat).mean()

    print(
        f"rand Residuals, div sigma = {rand_div_sigma_FT:4f}, equi err = {rand_equi_err:4f}, compat err = {rand_compat_err:4f}"
    )

    print(f"\t C homog FEA is {C_homog_FEA:4f}")

    MAX_ITERS = 10

    div_sigma_FT = torch.zeros(MAX_ITERS)
    equi_err = torch.zeros(MAX_ITERS)
    compat_err = torch.zeros(MAX_ITERS)
    C_homog = torch.zeros(MAX_ITERS)

    for i in range(MAX_ITERS):
        eps = G.forward(eps, C_field, use_polar=False)
        sigma = constlaw(C_field, eps)

        resid_equi, resid_compat = G.compute_residuals(eps, sigma)

        div_sigma_FT[i] = stressdiv(sigma).mean()
        equi_err[i] = constlaw.C0_norm(resid_equi).mean()
        compat_err[i] = constlaw.C0_norm(resid_compat).mean()
        C_homog[i] = sigma[0, 0].mean() / eps[0, 0].mean()

        if i % 1 == 0:
            print(
                f"Iter {i}, div sigma = {div_sigma_FT[i]:4f}, equi err = {equi_err[i]:4f}, compat err = {compat_err[i]:4f}"
            )
            print(f"\t C homog is {C_homog[i]:4f}")
            if i % 1 == 0:
                plot_cube(eps[0, 0], savedir=f"FFT_2phase_eps_{i}.png", cmap="coolwarm")
                plot_cube(
                    sigma[0, 0], savedir=f"FFT_2phase_sig_{i}.png", cmap="coolwarm"
                )
            # f = File(f"2phase_fft_{i}.h5", "w")
            # write_dataset_to_h5(C_field, "C_field", f)
            # write_dataset_to_h5(eps, "strain", f)

    import matplotlib.pyplot as plt

    print(abs(C_homog - C_homog_FEA))

    plot_cube(eps[0, 0], "fft_2phase_exx.png", cmap="coolwarm")
    plot_cube(sigma[0, 0], "fft_2phase_sxx.png", cmap="coolwarm")

    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    x = torch.arange(MAX_ITERS)
    ax[0].semilogy(x, div_sigma_FT)
    ax[0].set_title("Stress Divergence (FFT)")

    ax[1].semilogy(x, equi_err)
    ax[1].set_title("Equilibrium Error (C0)")

    ax[2].semilogy(x, compat_err)
    ax[2].set_title("Compatibility Error (C0)")

    ax[3].plot(x, abs(C_homog - C_homog_FEA))
    ax[3].set_title("C_homog error")

    fig.tight_layout()
    plt.savefig("FFT_convergence_trace.png", dpi=300)


def compare_convergences():
    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)

    def setup_model(checkpt_file, config_file=None):

        if checkpt_file is None:
            config = Config(**load_conf_override(config_file))
            model = make_localizer(config, constlaw)
        else:
            model = load_checkpoint(checkpt_file, strict=True)

        model.config.greens_upsample = 2
        model.config.num_voxels = 32
        model.overrideConstlaw(constlaw)
        model.config.therino_init_zero = True
        model.compute_scalings(E_BAR)

        return model

    # m_base = "paper2_32_hiCR"
    # r_base = "paper2_32_hiCR_u1_responses"

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"

    num_test = 100
    max_iters = 20
    num_iters_err = 16

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.TEST])
    dataset.assignConstlaw(constlaw)
    C_field, bc_vals, eps_FEA, sigma_FEA = dataset[
        PLOT_IND_BAD : PLOT_IND_BAD + num_test
    ]
    C_field = C_field.cuda()
    eps_FEA = eps_FEA.cuda()
    sigma_FEA = sigma_FEA.cuda()
    VM_FEA = VMStress(sigma_FEA)

    # C_field = constlaw.compute_C_field(m).cuda()
    bc_vals = bc_vals.cuda()

    # set up fig info
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    fig_2, ax_2 = plt.subplots(1, 2, figsize=(6, 3), sharex=True)
    fig_3, ax_3 = plt.subplots(1, 1, figsize=(5, 5))

    name_to_color = {
        "TherINO": "r",
        "FNO-DEQ": "g",
        "IFNO": "k",
        "Mod. F-D": "b",
        "FFT": "gray",
    }
    avg_2_norm = lambda x: (x**2).sum(dim=-4).mean((-3, -2, -1)).sqrt()

    def eval_model_convergence(model, label=None, is_deq=True):
        model = model.cuda()
        model = model.train()
        print(f"Evaluating {model.config.arch_str} convergence")
        with torch.inference_mode():
            iters = max_iters
            if isinstance(model, IFNOLocalizer):
                iters = 16
            elif isinstance(model, LocalizerFFT):
                iters = 64

            strain_trace, traj_trace, traj_resids = model._compute_trajectory(
                C_field,
                bc_vals,
                num_iters=iters + 1,
                return_latent=True,
                return_latent_resids=True,
            )

            strain_trace = torch.stack(strain_trace, dim=0).detach()
            traj_trace = torch.stack(traj_trace, dim=0).detach()

            strain_errs_full = torch.stack([eps - eps_FEA for eps in strain_trace[:-1]])
            # list of tensors of L2 strain errors relative to FEA
            strain_errs = avg_2_norm(model.scale_strain(strain_errs_full)).cpu()

            VM_scaling = mean_L1_error(VM_FEA, 0 * VM_FEA).cpu()

            VM_errs = torch.stack(
                [
                    (
                        mean_L1_error(
                            VMStress(model.constlaw(C_field, eps)), VM_FEA
                        ).cpu()
                        / VM_scaling
                    ).squeeze()
                    for eps in strain_trace
                ]
            ).cpu()

            diff_full = traj_trace[:-1] - traj_trace[-1]
            # take average 2 norm and divide by last iter RMS value
            diff = avg_2_norm(diff_full).cpu() / avg_2_norm(traj_trace[-1]).cpu()

            # DEQ: use residuals in latent space
            if is_deq:
                traj_resids = torch.stack(traj_resids, dim=0)
                traj_resids_norm = avg_2_norm(traj_resids[:-1]) / avg_2_norm(
                    traj_trace[:-1]
                )
                diff = traj_resids_norm.cpu()

            strain_diff_full = strain_trace[:-1] - strain_trace[-1]
            # strain_diff = (
            #     avg_2_norm(strain_diff_full).cpu() / avg_2_norm(traj_trace[-1]).cpu()
            # )

        # take mean and std over instance index (for all iterations)
        se_mean = strain_errs.mean(dim=1)
        (se_min, _), (se_max, _) = strain_errs.min(dim=1), strain_errs.max(dim=1)
        ve_mean = VM_errs.mean(dim=1)
        (ve_min, _), (ve_max, _) = VM_errs.min(dim=1), VM_errs.max(dim=1)
        diff_mean = diff.mean(dim=1)
        (diff_min, _), (diff_max, _) = diff.min(dim=1), diff.max(dim=1)

        def inner_prod(x, y):
            return torch.einsum("nbrijk, nbrijk -> nb", x, y)

        dot = inner_prod(strain_diff_full, strain_errs_full) / (
            inner_prod(strain_diff_full, strain_diff_full).sqrt()
            * inner_prod(strain_errs_full, strain_errs_full).sqrt()
            + 1e-6
        )
        # now get instance-averaged alignment
        dot = dot.mean(1).cpu()

        cos_vals = torch.arccos(dot).cpu()

        print("shapes", se_mean.shape, se_min.shape, se_max.shape)
        print(se_mean)

        xx = np.arange(iters) + 1

        ax_1.plot(
            xx,
            se_mean * 100.0,
            "-",
            c=name_to_color.get(label),
            label=label,
        )

        # scatter elapsed wall time vs. error
        # TODO get timing numbers for hybrid models
        ax_3.plot(
            TIMINGS.get(label, TIMINGS["TherINO"]) * xx,
            se_mean,
            c=name_to_color.get(label),
            marker=".",
            label=label,
        )
        # ax[0].fill_between(
        #     xx, se_min, se_max, label=label, alpha=0.3, color=name_to_color.get(label)
        # )

        # ax_1[1].plot(
        #     xx[:num_iters_err],
        #     ve_mean * 100.0,
        #     "-",
        #     c=name_to_color.get(label),
        #     label=label,
        # )
        # ax[1].fill_between(
        #     xx, ve_min, ve_max, label=label, alpha=0.3, color=name_to_color.get(label)
        # )

        if is_deq:
            ax_2[0].plot(xx, diff_mean, "-", c=name_to_color.get(label), label=label)

            ax_2[0].set_yscale("log")

            ax_2[1].plot(xx, dot, "-", c=name_to_color.get(label), label=label)

    eval_model_convergence(setup_model(CHECK_THERINO), "TherINO")

    eval_model_convergence(setup_model(CHECK_FNODEQ), "FNO-DEQ")

    eval_model_convergence(setup_model(CHECK_IFNO), "IFNO", is_deq=False)

    eval_model_convergence(setup_model(CHECK_THERNOTHER), "Mod. F-D", is_deq=True)

    # eval_model_convergence(setup_model(CHECK_FFT), "FFT", is_deq=False)
    # eval_model_convergence(setup_model(CHECK_THERPRE), "TherINO (Pre)", is_deq=True)
    # eval_model_convergence(setup_model(CHECK_THERPOST), "TherINO (Post)", is_deq=True)
    # eval_model_convergence(
    #     setup_model(CHECK_THERHYBRID), "TherINO (Hybrid)", is_deq=True
    # )

    ax_1.legend()
    ax_1.set_ylim(ymax=100)
    ax_1.set_xlabel("Iteration")
    ax_1.set_ylabel("L2 Strain Error (%)")
    # ax_1[1].set_title("L1 VM Stress Error (%)")
    # ax_1[1].set_ylim(ymax=150)
    # ax_1[1].xaxis.set_major_locator(tck.MultipleLocator())

    ax_2[0].legend()
    ax_2[0].set_title("DEQ Convergence")
    ax_2[1].set_title("DEQ Alignment")
    ax_2[0].set_ylabel("Residual")
    ax_2[1].set_ylabel("Alignment")
    ax_2[0].set_xlabel("Iteration")
    ax_2[1].set_xlabel("Iteration")

    ax_2[1].hlines(0, 0.1, max_iters - 0.1, colors="k", linestyles="--")
    # ax_2[1].xaxis.set_major_locator(tck.MultipleLocator())

    ax_3.scatter(0.388, 32.145 / 100.0, marker="v", label="FNO")
    ax_3.scatter(1.101, 19.638 / 100.0, marker="^", label="Big FNO")

    ax_3.set_yscale("log")

    ax_3.legend()
    ax_3.set_xlabel("Inference time (ms)")
    ax_3.set_ylabel("L2 Strain error")

    fig_1.tight_layout()
    fig_1.legend
    fig_1.savefig("convergence_errors.png", dpi=300)

    fig_2.tight_layout()
    fig_2.legend
    fig_2.savefig("convergence_resids.png", dpi=300)

    fig_3.tight_layout()
    fig_3.legend
    fig_3.savefig("tradeoff.png", dpi=300)


def compare_model_perf():
    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)

    NUM_ITERS = 12

    USE_CUDA = True

    def setup_model(checkpt_file, conf_file=None):
        if checkpt_file is None:
            config = Config(load_conf_override(CONF_THERINO))
            model = make_localizer(config, constlaw)
        else:
            model = load_checkpoint(checkpt_file, strict=True)
        model.config.num_voxels = 32
        model.config.deq_randomize_max = False
        model.config.deq_args["f_solver"] = "anderson"
        model.config.deq_args["f_max_iter"] = NUM_ITERS
        # extremely tight tolerance to avoid early return
        model.config.deq_args["f_tol"] = 1e-30
        model.config.num_ifno_iters = NUM_ITERS

        model.overrideConstlaw(constlaw)
        return model

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"
    UPSAMP_MICRO_FAC = 1

    num_test = 128 if USE_CUDA else 64

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.VALID])
    dataset.assignConstlaw(constlaw)
    C_field, bc_vals, _, _ = dataset[PLOT_IND_BAD : PLOT_IND_BAD + num_test]
    # eps_FEA = eps_FEA.cuda()
    # sigma_FEA = sigma_FEA.cuda()
    # VM_FEA = VMStress(sigma_FEA)

    if USE_CUDA:
        bc_vals = bc_vals.cuda()
        C_field = C_field.cuda()

    from torch.profiler import profile, record_function, ProfilerActivity

    # torch.set_num_threads(4)

    print(f"Running on {torch.get_num_threads()} threads")

    def eval_model_runtime(checkpt_file, config_file):
        model = setup_model(checkpt_file, config_file).eval()
        if USE_CUDA:
            model = model.cuda()

        print(f"Profiling model {model.config.arch_str}!")

        with torch.inference_mode():
            # warm-up runs

            model(C_field, bc_vals)

            # print("Warm up completed!")
            if USE_CUDA:
                torch.cuda.synchronize()

                with profile(
                    activities=([ProfilerActivity.CPU, ProfilerActivity.CUDA]),
                    with_stack=False,
                    record_shapes=False,
                ) as prof:
                    with record_function("model_inference"):
                        model(C_field, bc_vals)

                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=16
                    )
                )

                total_inference_time_ms = (1.0 / 1000.0) * sum(
                    [item.self_cuda_time_total for item in prof.key_averages()]
                )
            else:
                start = time.time()
                model(C_field, bc_vals)
                end = time.time()

                print("Total runtime is", end - start)

                total_inference_time_ms = (end - start) * 1000

            print(f"ms / micro: {total_inference_time_ms /  num_test}")
            # C_field = constlaw.compute_C_field(m).cuda()

    eval_model_runtime(CHECK_IFNO, CONF_IFNO)
    eval_model_runtime(CHECK_FNODEQ, CONF_FNODEQ)
    eval_model_runtime(CHECK_THERINO, CONF_THERINO)
    eval_model_runtime(CHECK_THERNOTHER, CONF_THERNOTHER)
    # eval_model_runtime(CHECK_THERHYBRID, CONF_THERHYBRID)
    # eval_model_runtime(CHECK_THERPRE, CONF_THERPRE)
    # eval_model_runtime(CHECK_THERPOST, CONF_THERPOST)


def test_iter_convergence():
    conf_args = load_conf_override(CONF_THERINO)

    config = Config(**conf_args)

    config.num_voxels = 32
    MAX_ITERS = 16 + 1

    config.add_resid_loss = False

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)
    model = make_localizer(config, constlaw)
    load_checkpoint(CHECK_THERINO, model, strict=True)

    # model = model.cuda()
    # required to get n_states to behave

    print(config)
    print(model)

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"
    UPSAMP_MICRO_FAC = 1

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.VALID])
    C_field, bc_vals, eps_FEA, sigma_FEA = dataset[PLOT_IND_BAD : PLOT_IND_BAD + 1]

    # C_field = model.constlaw.compute_C_field(m)
    # config.num_voxels = 32

    print("ITER C mean unnormalized", C_field[0].mean((-3, -2, -1)))
    print("C ref", constlaw.C_ref)
    print("C mean normalized", model.scale_stiffness(C_field[0].mean((-3, -2, -1))))

    VM_FEA = VMStress(sigma_FEA)

    base_pred = model.forward(C_field, bc_vals)

    print("Evaluating model")
    # don't store any gradients
    with torch.inference_mode():
        strain_trace = model._compute_trajectory(C_field, bc_vals, num_iters=MAX_ITERS)
    # print(strain_trace[0].shape)
    strain_trace = torch.stack(strain_trace, dim=0)

    last_pred = strain_trace[-1]
    last_pred = base_pred
    # print(strain_trace.shape)
    print("strain FEA", eps_FEA.mean((-3, -2, -1, 0)), eps_FEA.std((-3, -2, -1, 0)))
    print(
        "strain pred", last_pred.mean((-3, -2, -1, 0)), last_pred.std((-3, -2, -1, 0))
    )

    print("stress FEA", sigma_FEA.mean((-3, -2, -1, 0)), eps_FEA.std((-3, -2, -1, 0)))
    print(
        "stress pred",
        model.constlaw(C_field, last_pred).mean((-3, -2, -1, 0)),
        last_pred.std((-3, -2, -1, 0)),
    )

    print(last_pred[:, 0].squeeze().shape)
    plot_cube(last_pred[:, 0].squeeze(), "convergence_last.png")
    plot_cube(eps_FEA[:, 0, ...].squeeze(), "conv_last_FEA.png")

    print("Computing errors")
    # compute L1 errors as well
    strain_errs = [
        100
        * model.scale_strain(mean_L1_error(eps[0, 0], eps_FEA[0, 0])).cpu().squeeze()
        for eps in strain_trace
    ]

    VM_scaling = mean_L1_error(VM_FEA, 0 * VM_FEA).cpu().squeeze()

    VM_errs = [
        100
        * mean_L1_error(VMStress(model.constlaw(C_field, eps)), VM_FEA).cpu().squeeze()
        / VM_scaling
        for eps in strain_trace
    ]

    print(VM_scaling, VM_errs[0].shape)

    print("L1 Strain errors is", strain_errs)
    print("VM stress errors is", VM_errs)
    # sum difference along each component
    diff = strain_trace[:-1] - strain_trace[-1]
    # diff = torch.diff(strain_trace, dim=0) ** 2
    # take average L2 norm
    diff = (diff**2).sum(2).sqrt().mean((-3, -2, -1))

    print(diff.shape, diff)
    print(strain_errs[0].shape)

    print("plotting")
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(np.arange(MAX_ITERS - 1), strain_errs[:-1])
    ax[0].set_title("Percent Strain Error")
    # ax[0].set_yticks([1, 10, 20, 50],)
    # ax[0].get_yaxis().get_major_formatter().labelOnlyBase = False

    ax[1].plot(np.arange(MAX_ITERS - 1), VM_errs[:-1])
    ax[1].set_title("Percent VM Stress Error")
    # ax[1].set_yticks([1, 10, 20, 50],)
    # ax[1].get_yaxis().get_major_formatter().labelOnlyBase = False

    ax[2].semilogy(np.arange(MAX_ITERS - 1), diff)
    ax[2].set_title("DEQ Residual")
    ax[2].set_xlabel("Iteration")
    ax[2].xaxis.set_major_locator(tck.MultipleLocator())
    plt.tight_layout()

    plt.savefig("conv_trace.png", dpi=300)


def test_superres():
    pass


def test_solconv_pca():

    def autocorr(x):
        x = x.double()
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1), norm="forward")

        # N = x.shape[-3]
        # if N % 2 == 0:
        #     out_shape = [s - 1 for s in x.shape[-3:]]
        # else:
        out_shape = x.shape[-3:]
        s_ft = x_ft.conj() * x_ft
        s = torch.fft.ifftn(s_ft, dim=(-3, -2, -1), norm="forward", s=out_shape)

        print("s_imag", s.imag.max(), s.imag.abs().mean())

        s = s.real
        s = s.float()

        return s

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"
    UPSAMP_MICRO_FAC = 1

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)

    from main import dataset_info

    loader_fixed32 = load_data(dataset_info["fixed32"], DataMode.TEST, constlaw)
    loader_randcr32 = load_data(dataset_info["randcr32"], DataMode.TEST, constlaw)
    loader_randbc32 = load_data(dataset_info["randbc32"], DataMode.TEST, constlaw)
    loader_hiCR = load_data(dataset_info["hiCR32"], DataMode.TEST, constlaw)
    loader_poly = load_data(dataset_info["poly64"], DataMode.TEST, constlaw)

    # bc_vals = bc_vals.cuda()

    from pca import TorchPCA

    cutoff = -1

    def get_u_auto(dataset, name=""):
        _C_field, _bc_vals, _eps, _sigma = dataset[:cutoff]

        # eps_equiv = deviatoric(_eps, mandel_form=True)
        # sigma_equiv = deviatoric(_sigma, mandel_form=True)
        energy = compute_strain_energy(_eps, _sigma).cuda()

        total_energy = energy.mean((-3, -2, -1))

        print(
            f"{total_energy.min():2.4f}, {total_energy.max():2.4f}, {total_energy.mean():2.4f}, {total_energy.std():2.4f}"
        )
        # divide out total energy
        sqrt_u = energy.sqrt()
        u_auto = autocorr(sqrt_u)

        print(u_auto.shape, sqrt_u.shape)

        plot_cube(_eps[0:1, 0], f"{name}_eps_FEA.png")
        plot_cube(_sigma[0:1, 0], f"{name}_sig_FEA.png")

        plot_cube(energy[0:1, 0], f"{name}_energy_FEA.png")
        plot_cube(torch.fft.fftshift(u_auto[0:1, 0], (-2, -3)), f"{name}_auto_FEA.png")

        return u_auto

    def get_C_homog(dataset):
        _C_field, _bc_vals, _eps, _sigma = dataset[:cutoff]
        return est_homog(_eps, _sigma, (0, 0)).cpu().squeeze().numpy()

    import umap

    PCA = TorchPCA(k=1024, center=True, whiten=False).cuda()
    reducer = umap.UMAP(min_dist=0.2, n_neighbors=40)

    def pipeline(dataset, fit=False, name=""):
        print("Computing autocorrs")
        u_auto = get_u_auto(dataset, name=name)

        if fit:
            # fit PCA now
            PCA.fit(u_auto)

            print(PCA.relative_variance[:10])
            pcs = PCA(u_auto).cpu().numpy()

            # fit PC space
            reducer.fit(pcs)
            umap_scores = reducer.transform(pcs)

        else:
            pcs = PCA(u_auto).cpu().numpy()
            umap_scores = reducer.transform(pcs)

        C_homog = get_C_homog(dataset)

        return umap_scores, pcs, C_homog

    umap_scores_bc32, pcs_bc32, C_homog_bc32 = pipeline(
        loader_randbc32.dataset, fit=True, name="randbc32"
    )
    umap_scores_cr32, pcs_cr32, C_homog_cr32 = pipeline(
        loader_randcr32.dataset, name="randcr32"
    )
    umap_scores_f32, pcs_f32, C_homog_f32 = pipeline(
        loader_fixed32.dataset, name="fixed32"
    )
    umap_scores_hiCR, pcs_hiCR, C_homog_hiCR = pipeline(
        loader_hiCR.dataset, name="hiCR"
    )
    umap_scores_poly, pcs_poly, C_homog_poly = pipeline(
        loader_poly.dataset, name="poly"
    )

    plt.figure()
    plt.scatter(
        pcs_cr32[:, 0], pcs_cr32[:, 1], marker=".", label="Random Contrast", alpha=0.5
    )
    plt.scatter(
        pcs_hiCR[:, 0], pcs_hiCR[:, 1], marker="v", label="Highest Contrast", alpha=0.5
    )
    # plt.scatter(pcs_bc32[:, 0], pcs_bc32[:, 1], marker="^", label="randbc32", alpha=0.5)
    plt.scatter(pcs_poly[:, 0], pcs_poly[:, 1], marker="x", label="Cubic", alpha=0.5)
    # plt.scatter(pcs_f32[:, 0], pcs_f32[:, 1], marker=".", label="fixed32", alpha=0.5)
    plt.legend()

    plt.xlabel("PC1"), plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("energy_pca.png", dpi=300)

    plt.figure()
    plt.scatter(
        umap_scores_hiCR[:, 0],
        umap_scores_hiCR[:, 1],
        marker="v",
        label="hiCR",
        alpha=0.5,
    )
    plt.scatter(
        umap_scores_bc32[:, 0],
        umap_scores_bc32[:, 1],
        marker="^",
        label="randbc32",
        alpha=0.5,
    )
    plt.scatter(
        umap_scores_cr32[:, 0],
        umap_scores_cr32[:, 1],
        marker="x",
        label="randcr32",
        alpha=0.5,
    )
    plt.scatter(
        umap_scores_f32[:, 0],
        umap_scores_f32[:, 1],
        marker=".",
        label="fixed32",
        alpha=0.5,
    )
    plt.scatter(
        umap_scores_poly[:, 0],
        umap_scores_poly[:, 1],
        marker="o",
        label="poly",
        alpha=0.5,
    )
    plt.legend()

    plt.xlabel("UMAP1"), plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig("energy_umap.png", dpi=300)

    return

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS).cuda()

    def setup_model(config_file, checkpt_file):
        model = load_checkpoint(checkpt_file, strict=True)
        model.config.num_voxels = 32
        # start with zero guess
        model.config.therino_init_zero = True
        model.overrideConstlaw(constlaw)
        return model

    def get_pc_traj(model):
        model = model.cuda()
        model = model.eval()

        print(f"Evaluating {model.config.arch_str} convergence")
        with torch.inference_mode():
            strain_trace = model._compute_trajectory(
                C_field[0:1], bc_vals[0:1], num_iters=12
            )

            strain_trace = torch.stack(strain_trace, dim=0)
            stress_trace = strain_to_stress(C_field[0:1], strain_trace)

            energy_trace = compute_strain_energy(stress_trace, strain_trace)

            print(strain_trace.shape)

            plot_cube(eps_FEA[0:1, 0].squeeze(), "eps_FEA_pca.png")
            plot_cube(strain_trace[-1, 0, 0].squeeze(), "eps_pred_pca.png")
            plot_cube(sigma_FEA[0:1, 0].squeeze(), "sig_FEA_pca.png")
            plot_cube(stress_trace[-1, 0, 0].squeeze(), "sig_pred_pca.png")

            plot_cube(energy[0:1, 0].squeeze(), "energy_FEA_pca.png")
            plot_cube(energy_trace[-1, 0, 0].squeeze(), "energy_pred_pca.png")

            print(energy_trace.shape)
            pc_trace = pca.forward(autocorr(energy_trace.sqrt()))
            return pc_trace.cpu().numpy()

    pc_trace_ifno = get_pc_traj(setup_model(CONF_IFNO, CHECK_IFNO))
    pc_trace_fno_deq = get_pc_traj(setup_model(CONF_FNODEQ, CHECK_FNODEQ))
    pc_trace_therino = get_pc_traj(setup_model(CONF_THERINO, CHECK_THERINO))

    plt.figure()
    plt.scatter(pcs[0, 0], pcs[0, 1], marker="x", label="FEA")
    # plt.scatter(
    #     pcs[:, 0], pcs[:, 1], alpha=0.05, c="gray", marker=".", label="training set"
    # )

    def plot_trace(trace, name, marker):
        # gets darker as we converge
        colors = np.linspace(0.5, 1, trace.shape[0])
        plt.scatter(
            trace[:, 0],
            trace[:, 1],
            c=colors,
            label=f"{name} traj",
            alpha=0.5,
            marker=marker,
            cmap="gray_r",
        )

    plot_trace(pc_trace_therino, "TherINO", "o")
    # plot_trace(pc_trace_ifno, "IFNO", "v")
    plot_trace(pc_trace_fno_deq, "FNO-DEQ", "^")

    leg = plt.legend()
    # set all to black in legend
    [leg.legend_handles[i].set_color("k") for i in range(1, len(leg.legend_handles))]
    plt.savefig("pc_trace.png", dpi=300)


def test_compare_janus(N=32, willot=False, fac=1):
    # compare our version of green's op w/ Janus
    import numpy as np

    import janus
    import janus.material.elastic.linear.isotropic as material
    import janus.operators as operators
    import janus.fft.serial as fft
    import janus.green as green

    # young's modulus
    E = 2
    # poisson ratio
    nu = 0

    # get shear modulus
    _, mu = YMP_to_Lame(E, nu)

    shape = (N, N, N)
    shape_fac = (N * fac, N * fac, N * fac)

    C0 = material.create(mu, nu, dim=3)
    G0 = C0.green_operator()

    G_janus = green.truncated(
        G0, shape_fac, 1.0 / (N * fac), fft.create_real(shape_fac)
    )
    if willot:
        G_janus = green.willot2015(
            G0, shape_fac, 1.0 / (N * fac), fft.create_real(shape_fac)
        )

    def apply_janus(tau):
        tau = upsample_field(tau, fac)
        tau_orig_shape = tuple(tau.shape)
        tau = tau.squeeze()
        # swap tensor index to end
        tau = tau.permute(1, 2, 3, 0).double().numpy()

        out = np.zeros_like(tau)
        G_janus.apply(tau, out)
        # convert output to torch, then fix shape and batch
        out = torch.from_numpy(out).permute(3, 0, 1, 2).reshape(tau_orig_shape)
        out = average_field(out, fac)

        print("out", out.shape, tau_orig_shape)
        return out

    constlaw = StrainToStress_2phase([E, E], [nu, nu]).double()
    constlaw_crystal = StrainToStress_crystal(C11, C12, C44).double()
    greens_op = GreensOp(
        constlaw,
        N * fac,
        willot=willot,
        nyquist_method=NyquistMethod.STRESS,
    ).double()
    greens_op.G_freq = greens_op.G_freq.cdouble()

    print(constlaw.C_ref)
    print(C0)

    np.seterr(all="warn", over="raise")

    # # get euler ang and stiffness tens
    # f = File("01_CubicSingleEquiaxedOut.dream3d")

    # euler_ang = f["DataContainers"]["SyntheticVolumeDataContainer"]["CellData"][
    #     "EulerAngles"
    # ][:]
    # euler_ang = torch.from_numpy(euler_ang)

    # C_field = constlaw_crystal.compute_C_field(euler_ang[None].double())

    # e_0 = torch.ones(1, 6, N, N, N).double()
    # tau = strain_to_stress(C_field, e_0)

    tau = torch.zeros(1, 6, N, N, N).double()

    tau[..., N // 2 :, N // 4 :, N // 8 :] = 1
    tau[..., 0, 0, 0] = 1

    # do averaging inside both models
    tau_jan = apply_janus(tau)
    tau_torch = greens_op.apply_gamma(upsample_field(tau, fac))
    tau_torch = average_field(tau_torch, fac)

    norm = tau_jan.abs().mean()
    if norm <= 1e-6:
        norm = 1

    diff = (tau_jan - tau_torch).abs() / norm

    if False:
        print("abs diff", diff.mean(), diff.std())

        print("max diff ind is", np.unravel_index(torch.argmax(diff), diff.shape))

        def transform(f, real=False):
            # fft and shift
            f = torch.fft.fftshift(torch.fft.fftn(f, dim=(-3, -2, -1)))
            if real:
                f = f.real
            return f.abs().sum(1)

        plot_cube(
            equivalent(tau_jan).squeeze() / norm, savedir=f"stencil_janus_{willot}.png"
        )
        plot_cube(
            equivalent(tau_torch).squeeze() / norm,
            savedir=f"stencil_custom_{willot}.png",
        )
        plot_cube(
            (equivalent(tau_jan) - equivalent(tau_torch).squeeze()) / norm,
            savedir=f"stencil_diff_{willot}.png",
        )

        tfj = torch.fft.fftshift(torch.fft.fftn(tau_jan, dim=(-3, -2, -1)))
        tft = torch.fft.fftshift(torch.fft.fftn(tau_torch, dim=(-3, -2, -1)))

        # compare ratios
        print("real", tfj[0, 0, :, 0, -1].real, tft[0, 0, :, 0, -1].real)
        print("imag", tfj[0, 0, :, 0, -1].imag, tft[0, 0, :, 0, -1].imag)

        plt.figure()
        plt.plot(np.arange(N), transform(tau_jan)[0, :, 0, -1], "x", label="janus")
        plt.plot(np.arange(N), transform(tau_torch)[0, :, 0, -1], label="custom")
        plt.savefig(f"fourier_comp_{willot}.png", dpi=300)

        plot_cube(transform(tau_jan), savedir=f"fourier_janus_{willot}.png")
        plot_cube(transform(tau_torch), savedir=f"fourier_custom_{willot}.png")
        plot_cube(
            transform(tau_jan - tau_torch),
            savedir=f"fourier_diff_{willot}.png",
        )

        plt.plot(np.arange(N), equivalent(tau_jan)[0, :, 0, 0], "x", label="janus")
        plt.plot(np.arange(N), equivalent(tau_torch)[0, :, 0, 0], label="custom")

        plt.legend()
        plt.savefig(f"stencil_comp_{willot}.png", dpi=300)

        jan_freqs = torch.zeros(6, 6, N * fac, N * fac, N * fac)

        print("Computing janus freqs")
        for i, j, k in itertools.product(torch.arange(N * fac), repeat=3):
            G_janus.set_frequency(np.array([i, j, k], dtype=np.int32))

            jan_freqs[:, :, i, j, k] = torch.from_numpy(
                np.asarray(G_janus.to_memoryview())
            )

        freq_diff = (greens_op.G_freq - jan_freqs).abs()
        print("freq diff", freq_diff.mean((-3, -2, -1)), freq_diff.max())
        mfi = np.unravel_index(torch.argmax(freq_diff), freq_diff.shape)
        print("max freq ind is", mfi)

        ps = np.s_[..., mfi[-2], mfi[-1]]

        plt.figure()
        plt.plot(np.arange(N * fac), jan_freqs[3, 4][ps].real, "x", label="janus")
        plt.plot(np.arange(N * fac), greens_op.G_freq[3, 4][ps].real, label="custom")

        plt.legend()
        plt.savefig(f"freqs_comp_{willot}_xs.png", dpi=300)

        plt.figure()
        plt.plot(np.arange(N * fac), jan_freqs[0, 1][ps].real, "x", label="janus")
        plt.plot(np.arange(N * fac), greens_op.G_freq[0, 1][ps].real, label="custom")

        plt.legend()
        plt.savefig(f"freqs_comp_{willot}_xy.png", dpi=300)

        print("Plotting")
        plot_cube(jan_freqs[0, 0].real, savedir="janus_frequencies_xx.png")
        plot_cube(greens_op.G_freq[0, 0].real, savedir="custom_frequencies_xx.png")
        plot_cube(
            greens_op.G_freq[0, 0].real - jan_freqs[0, 0].real,
            savedir="freq_diff_xx.png",
        )

        plot_cube(jan_freqs[0, 1].real, savedir="janus_frequencies_xy.png")
        plot_cube(greens_op.G_freq[0, 1].real, savedir="custom_frequencies_xy.png")
        plot_cube(
            greens_op.G_freq[0, 1].real - jan_freqs[0, 1].real,
            savedir="freq_diff_xy.png",
        )

        plot_cube(jan_freqs[0, 3].real, savedir="janus_frequencies_xs.png")
        plot_cube(greens_op.G_freq[0, 3].real, savedir="custom_frequencies_xs.png")
        plot_cube(
            greens_op.G_freq[0, 3].real - jan_freqs[0, 3].real,
            savedir="freq_diff_xs.png",
        )

    assert torch.allclose(tau_jan, tau_torch, atol=1e-6)


def save_dummy_fft_model():

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)
    g_solver = LocalizerFFT(Config(**load_conf_override(CONF_FFT)), constlaw)
    g_solver.overrideConstlaw(constlaw)
    g_solver.compute_scalings(E_BAR)

    save_checkpoint(
        g_solver, None, None, 0, 0, path_override=f"{CHECKPOINT_DIR}/fft.ckpt"
    )


def test_pooling():
    f = torch.randn(32, 6, 32, 32, 32)
    f_avg = average_field(f, 2)
    print(f_avg.shape)

    f_up = upsample_field(f, 2)
    f_up_down = average_field(f_up, 2)
    print(f_up.shape)
    print(f_up_down.shape)
    assert torch.allclose(f_up_down, f)

    f_up = upsample_field(f, 4)
    f_up_down = average_field(f_up, 4)
    assert torch.allclose(f_up_down, f)


# compare_convergences()
# test_solconv_pca()
test_euler_pred(CONF_THERHYBRID, CHECK_THERHYBRID)
test_euler_pred(CONF_THERPRE, CHECK_THERPRE)
test_euler_pred(CONF_THERPOST, CHECK_THERPOST)

test_euler_pred(CONF_FFT, CHECK_FFT)

test_euler_pred(CONF_FNODEQ, CHECK_FNODEQ)
test_euler_pred(CONF_IFNO, CHECK_IFNO)
test_euler_pred(CONF_FF, CHECK_FF)
test_euler_pred(CONF_THERINO, CHECK_THERINO)
exit()
# test_pooling()

# test_euler_pred(CONF_FFT, CHECK_FFT)
test_compare_janus(N=31, willot=False)
test_compare_janus(N=31, willot=True)
# upsample should get rid of high-freq difference
test_compare_janus(N=32, willot=False, fac=2)
test_compare_janus(N=32, willot=True, fac=2)

# save_dummy_fft_model()


test_FFT_iters_2phase()
test_FFT_iters_crystal()

# test_euler_ang()
# test_stiff_ref()


# test_constlaw_2phase()
# test_constlaw_cubic()
# test_mandel()
# test_rotat()
# test_mat_vec_op()


# test_euler_pred(CONF_THERHYBRID, CHECK_THERHYBRID)
# test_euler_pred(CONF_THERPRE, CHECK_THERPRE)
# test_euler_pred(CONF_THERNOTHER, CHECK_THERNOTHER)


# compare_model_perf()

# test_model_save_load()
# # make_error_plot()
# test_iter_convergence()

# test_varying_datasets()
# # test_super_res()
# prof_C_op()


# test_fft_deriv()
