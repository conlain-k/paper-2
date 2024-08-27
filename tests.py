import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch

from constlaw import *
from helpers import *
from tensor_ops import *
from plot_cube import *
from euler_ang import *
from greens_op import *
from loaders import *
from config import Config
from solvers import make_localizer
import time
from h5py import File

from main import load_data

import numpy as np
import matplotlib.ticker as tck


PLOT_IND_BAD = 1744

BS = 32
N = 31
k = 2 * PI / N

E_VALS = [120.0, 100 * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = torch.as_tensor([0.001, 0, 0, 0, 0, 0]).reshape(1,6,1,1,1)

C11, C12, C44 = 200, 100, 200

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

CHECK_FNODEQ ="checkpoints/model_fno_deq_18.9M_s32_best.ckpt"
CONF_FNODEQ = "configs/fno_deq.json"

CHECK_THERMINO = "checkpoints/model_thermino_18.9M_s32_best.ckpt"
CONF_THERMINO = "configs/thermino.json"

CHECK_THERNOTHER = "checkpoints/model_thermino_notherm_18.9M_s32_best.ckpt"
CONF_THERNOTHER = "configs/thermino_notherm.json"

CHECK_IFNO = "checkpoints/model_ifno_18.9M_s32_best.ckpt"
CONF_IFNO = "configs/ifno.json"

CHECK_TEST = CHECK_THERMINO
CONF_TEST = CONF_THERMINO


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


def test_constlaw():
    CR_str = "100.0"
    m_base = "paper2_smooth"

    datasets, CR = collect_datasets("paper2_smooth", 100.0)
    dataset = LocalizationDataset(**datasets[DataMode.TRAIN])

    E_VALS = [120.0, CR * 120.0]
    NU_VALS = [0.3, 0.3]

    print(E_VALS, NU_VALS)

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)

    inds = [0, 5, 120, 690, 1111]
    micro,_, strain, stress = dataset[inds]
    check_constlaw(constlaw, micro, strain, stress)


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
    print(f"Simultaneous time is {time_simul:.3f}, premul time is {time_premul:.3f}")


def prof_C_op():
    # profile stress computation using C and m

    micros = torch.randn(128, 2, 31, 31, 31).cuda()
    C_op = StrainToStress_2phase([1, 1000], [0.3, 0.3]).cuda()
    strains = torch.randn(128, 6, 31, 31, 31).cuda()

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

    # print(euler_ang.shape)

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


def test_euler_pred():

    f = File("01_CubicSingleEquiaxedOut.dream3d")

    euler_ang = f["DataContainers"]["SyntheticVolumeDataContainer"]["CellData"][
        "EulerAngles"
    ][:]

    Z = 2 * C44 / (C11 - C12)

    print(f"Zener ratio is {Z:.2f}")

    euler_ang = torch.from_numpy(euler_ang)
    # euler_ang = euler_ang.cuda()

    conf_args = load_conf_override(CONF_TEST)
    # conf_args["fno_args"]["modes"] = (16, 16)
    # print(conf_args)
    config = Config(**conf_args)
    config.num_voxels = 32

    # no need to return residual
    config.add_resid_loss = False
    # config.fno_args["normalize_inputs"] = False
    # config.fno_args["use_mlp_lifting"] = False

    model = make_localizer(config)

    constlaw = StrainToStress_crystal(C11, C12, C44, E_BAR)
    model.setConstlaw(constlaw)
    # model.greens_op = GreensOp(model.constlaw, 62)

    # override to set crystalline const law
    # pull in saved weights
    model = load_checkpoint(CHECK_TEST, model, strict=True)

    # make sure we're ready to eval
    # model = model.cuda()
    model.eval()

    import time

    print("running inference!")

    C_field = model.constlaw.compute_C_field(euler_ang[None])

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        # evaluate on batch of euler angles
        strain_pred = model.forward(C_field, E_BAR.reshape((-1,6,1,1,1)))

    torch.cuda.synchronize()
    dt = time.time() - start

    # print(strain_pred.shape)

    print("elapsed time", dt)

    stress_pred = model.constlaw(C_field, strain_pred)

    homog_pred = est_homog(strain_pred, stress_pred, (0, 0)).squeeze()

    # now compare to true solution
    true_resp_file = "/storage/home/hcoda1/3/ckelly84/scratch/poly_stress_strain.h5"

    with h5py.File(true_resp_file, "r") as f:
        strain_true = (
            torch.as_tensor(f["strain"][:])
            .reshape(strain_pred.shape)
            .to(strain_pred.device)
        )
        stress_true = (
            torch.as_tensor(f["stress"][:])
            .reshape(stress_pred.shape)
            .to(stress_pred.device)
        ) / 10.0
        homog_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()

    print("homog shape", homog_true.shape)

    print(
        f"\nHomog true: {homog_true:4f} pred: {homog_pred:4f} rel_err: {(homog_true - homog_pred)/homog_true:4f}"
    )

    L1_strain_err = (strain_pred - strain_true).abs()[0, 0]
    # get percent error
    L1_strain_err = L1_strain_err * 100 / 0.001
    L1_stress_err = (stress_pred - stress_true).abs()[0, 0]

    plot_cube(
        L1_strain_err.detach().cpu(),
        "poly_extrap_exx_err.png",
        cmap="coolwarm",
        title="Percent L1 strain error",
    )

    plot_cube(
        L1_stress_err.detach().cpu(),
        "poly_extrap_sxx_err.png",
        cmap="coolwarm",
    )

    # strain_11_min = min(strain_pred[0, 0].min(), strain_true[0, 0].min())
    # strain_11_max = max(strain_pred[0, 0].max(), strain_true[0, 0].max())

    # stress_11_min = min(stress_pred[0, 0].min(), stress_true[0, 0].min())
    # stress_11_max = max(stress_pred[0, 0].max(), stress_true[0, 0].max())

    strain_11_min = strain_true[0, 0].min()
    strain_11_max = strain_true[0, 0].max()

    stress_11_min = stress_true[0, 0].min()
    stress_11_max = stress_true[0, 0].max()

    plot_cube(
        strain_pred[0, 0, :, :, :].detach().cpu(),
        "poly_extrap_exx.png",
        cmap="coolwarm",
        # vmin=strain_11_min,
        # vmax=strain_11_max,
    )
    plot_cube(
        stress_pred[0, 0, :, :, :].detach().cpu(),
        "poly_extrap_sxx.png",
        cmap="coolwarm",
        # vmin=stress_11_min,
        # vmax=stress_11_max,
    )

    plot_cube(
        strain_true[0, 0, :, :, :].detach().cpu(),
        "poly_true_exx.png",
        cmap="coolwarm",
        vmin=strain_11_min,
        vmax=strain_11_max,
    )
    plot_cube(
        stress_true[0, 0, :, :, :].detach().cpu(),
        "poly_true_sxx.png",
        cmap="coolwarm",
        vmin=stress_11_min,
        vmax=stress_11_max,
    )

    plot_cube(C_field[0, 0, 0, :, :, :].detach().cpu(), "d3d_C11.png", cmap="coolwarm")

    plot_cube(
        euler_ang[..., 0].detach().cpu(),
        "euler_ang.png",
        cmap="coolwarm",
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

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS, E_BAR)


    # make first model
    model = make_localizer(config)
    model = load_checkpoint(CHECK_TEST, model, strict=True)
    model.setConstlaw(constlaw)
    model.eval()
    model.cuda()

    TEST_PATH = "checkpoints/test.ckpt"

    # save these (randomly-init) params
    save_checkpoint(model, None, None, 0, 0, path_override=TEST_PATH, backup_prev=False)

    # now try loading the re-saved model
    model_2 = make_localizer(config)
    model_2=  load_checkpoint(TEST_PATH, model_2, strict=True)
    model_2.setConstlaw(constlaw)
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

    # print(euler_ang.shape)

    constlaw = StrainToStress_crystal(C11, C12, C44)

    C_field = constlaw.compute_C_field(euler_ang[None])

    G = GreensOp(constlaw, 62)

    eps = torch.zeros(1, 6, 62, 62, 62)
    eps[:, 0] = 0.001

    import matplotlib as mpl

    mpl.rcParams["figure.facecolor"] = "white"

    # now compare to true solution
    true_resp_file = "/storage/home/hcoda1/3/ckelly84/scratch/poly_stress_strain.h5"

    with h5py.File(true_resp_file, "r") as f:
        strain_true = torch.as_tensor(f["strain"][:]).reshape(eps.shape).to(eps.device)
        stress_true = torch.as_tensor(f["stress"][:]).reshape(eps.shape).to(eps.device)
        homog_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()

    print(f"True Cstar {homog_true:4f}")

    if torch.cuda.is_available():
        C_field = C_field.cuda()
        eps = eps.cuda()
        constlaw = constlaw.cuda()
        G = G.cuda()

    eps_0 = eps
    for i in range(10):
        eps = G.forward(eps, C_field)
        sig = constlaw(C_field, eps)
        equib_err, compat_err = G.compute_residuals(eps, sig)
        if i % 1 == 0:
            Cstar = est_homog(eps, sig, (0, 0)).squeeze()
            print(
                f"Iter {i} Cstar: {Cstar:4f}, equib: {constlaw.C0_norm(equib_err).mean()} compat {constlaw.C0_norm(compat_err).mean()}"
            )
            plot_cube(
                eps[0, 0].detach().cpu(), savedir=f"FFT_eps_{i}.png", cmap="coolwarm"
            )
        # f = File(f"crystal_fft_{i}.h5", "w")
        # write_dataset_to_h5(C_field, "C_field", f)
        # write_dataset_to_h5(eps, "strain", f)

    # print(constlaw.C0_norm(equib_err)[0].shape)
    # print(constlaw.C0_norm(compat_err)[0].shape)


def test_FFT_multi_2phase():
    def get_central_inds(size_new, size_old):
        # get a slice corresponding to the center of a volume
        # get new stencil sizes in each direction
        s1, s2 = size_new[-3], size_new[-2]

        # get midpoints of new weights tensor
        m1, m2 = s1 // 2, s2 // 2

        print(m1, m2, s1, s2)

        # input is biijk, output is boijk, elem-wise multiply along last
        # go from center - m to center + m (not including endpoints)
        inds = np.s_[
            :,
            :,
            m1 - size_old[0] // 2 + 1 : m1 + size_old[0] // 2,
            m2 - size_old[1] // 2 + 1 : m2 + size_old[1] // 2,
            0 : size_old[2],
        ]

        return inds

    def trig_interp(field_old, fac):
        field_ft = torch.fft.fftshift(
            torch.fft.fftn(field_old, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        )
        nx, ny, nz = field_ft.shape[-3:]
        new_shape = field_ft.shape[0:-3] + (fac * nx, fac * ny, fac * nz)
        field_ft_new = field_ft.new_zeros(new_shape)
        # get central inds to copy into
        inds = get_central_inds(field_ft_new.shape[-3:], field_ft.shape[-3:])
        # copy old signal into center
        field_ft_new[inds] = field_ft
        # now unshift so that origin is in corner again
        field_ft_new = torch.fft.ifftshift(field_ft_new)

        return field_ft_new

    m_base = "paper2_16"
    r_base_16 = "paper2_16_u1_responses"
    r_base_32 = "paper2_16_u2_responses"

    datasets_16, _ = collect_datasets(m_base, 100.0, r_base=r_base_16)
    # also get upsampled version for comparison
    datasets_32, _ = collect_datasets(
        m_base, 100.0, r_base=r_base_32, upsamp_micro_fac=2
    )

    dataset_16 = LocalizationDataset(**datasets_16[DataMode.TRAIN])
    dataset_32 = LocalizationDataset(**datasets_16[DataMode.TRAIN])
    m_16, eps_FEA_16, sigma_FEA_16 = dataset_16[0:1]
    m_32, eps_FEA_32, sigma_FEA_32 = dataset_32[0:1]

    m_16 = torch.as_tensor(m_16)
    eps_FEA_16 = torch.as_tensor(eps_FEA_16)
    sigma_FEA_16 = torch.as_tensor(sigma_FEA_16)

    m_32 = torch.as_tensor(m_32)
    eps_FEA_32 = torch.as_tensor(eps_FEA_32)
    sigma_FEA_32 = torch.as_tensor(sigma_FEA_32)

    constlaw = StrainToStress_2phase([120, 120 * 100], [0.3, 0.3])

    C_field_16 = constlaw.compute_C_field(m_16)

    G_16 = GreensOp(constlaw, 16)
    G_32 = GreensOp(constlaw, 32)

    def L2_strain_err(eps):
        # get 2-norm of strain error on 32x32 level (assuming already upsampled)
        # average L2 norm
        return ((eps - eps_FEA_32) ** 2).sum(1).sqrt().mean()

    def run_fft_iters(G, C_field, eps_0, max_it=10):
        eps = eps_0
        resids_equi = [None] * max_it
        resids_compat = [None] * max_it
        for i in range(max_it):
            eps = G.forward(eps, C_field)
            sigma = constlaw(C_field, eps)

            if eps.shape != eps_FEA_32.shape:
                eps_upsamp = trig_interp(eps)

            resid_equi, resid_compat = G.compute_residuals(eps, sigma)

    ax[1].semilogy(x, equi_err)
    ax[1].set_title("Equilibrium Error (C0)")

    ax[2].semilogy(x, compat_err)
    ax[2].set_title("Compatibility Error (C0)")

    ax[3].plot(x, abs(C_homog - C_homog_FEA))
    ax[3].set_title("C_homog error")

    fig.tight_layout()
    plt.savefig("FFT_convergence_trace.png", dpi=300)


def test_FFT_iters_2phase():

    m_base = "paper2_smooth"
    r_base = None

    m_base = "paper2_16"
    r_base = "paper2_16_u1_responses"
    # UPSAMP_MICRO_FAC = 1

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.TRAIN])
    m,bc_vals, eps_FEA, sigma_FEA = dataset[0:1]

    # resp_f = h5py.File("/storage/home/hcoda1/3/ckelly84/scratch/outputs/spher_inc_cr100.0_bc0/00000.h5")
    # eps_FEA, sigma_FEA = resp_f["strain"][:], resp_f["stress"][:]

    # resp_f.close()

    # m = h5py.File("/storage/home/hcoda1/3/ckelly84/scratch/micros/spher_inc.h5")["micros"][:]

    m = torch.as_tensor(m)
    eps_FEA = torch.as_tensor(eps_FEA)
    sigma_FEA = torch.as_tensor(sigma_FEA)

    UPSAMP = 1

    m = upsample_field(m, UPSAMP)
    eps_FEA = upsample_field(eps_FEA, UPSAMP)
    sigma_FEA = upsample_field(sigma_FEA, UPSAMP)

    N = m.shape[-1]

    constlaw = StrainToStress_2phase([120, 120 * 100], [0.3, 0.3])

    C_field = constlaw.compute_C_field(m)

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

    MAX_ITERS = 50

    div_sigma_FT = torch.zeros(MAX_ITERS)
    equi_err = torch.zeros(MAX_ITERS)
    compat_err = torch.zeros(MAX_ITERS)
    C_homog = torch.zeros(MAX_ITERS)

    for i in range(MAX_ITERS):
        eps = G.forward(eps, C_field)
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
            # plot_cube(eps[0, 0], savedir=f"FFT_2phase_eps_{i}.png", cmap="coolwarm")
            # plot_cube(sigma[0, 0], savedir=f"FFT_2phase_sig_{i}.png", cmap="coolwarm")
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
    def setup_model(checkpt_file, config_file):
        conf_args = load_conf_override(config_file)
        config = Config(**conf_args)
        config.num_voxels = 32
        # config.deq_args["f_solver"] = "fixed_point_iter"


        model = make_localizer(config)
        model.setConstlaw(constlaw)
        load_checkpoint(checkpt_file, model, strict=True)
        return model
    

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS, E_BAR) 

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"
    UPSAMP_MICRO_FAC = 1

    num_test = 100
    max_iters = 16 + 1

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.VALID])
    m, bc_vals, eps_FEA, sigma_FEA = dataset[PLOT_IND_BAD : PLOT_IND_BAD + num_test]
    eps_FEA = eps_FEA.cuda()
    sigma_FEA = sigma_FEA.cuda()
    VM_FEA = VMStress(sigma_FEA)

    C_field = constlaw.compute_C_field(m).cuda()
    bc_vals = bc_vals.cuda()

    # set up fig info
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    name_to_color = {"thermino": "r", "fno_deq": "g", "ifno":"b"}
    
    def eval_model_convergence(model, label=None):
        model = model.cuda()
        model = model.train()
        print(f"Evaluating {model.config.arch_str} convergence")
        with torch.inference_mode():
            strain_trace = model._compute_trajectory(C_field, bc_vals, num_iters=max_iters)

            strain_trace = torch.stack(strain_trace, dim=0)

            last_pred = strain_trace[-1]
            # list of tensors of L2 strain errors relative to FEA
            strain_errs = torch.stack([
                100 * ((eps - eps_FEA)**2).sum(dim=1).mean(dim=(-3,-2,-1)).cpu()
                / model.constlaw.strain_scaling
                for eps in strain_trace
            ]).cpu()

            VM_scaling = mean_L1_error(VM_FEA, 0 * VM_FEA).cpu()

            VM_errs = torch.stack([
                100
                * (mean_L1_error(VMStress(model.constlaw(C_field, eps)), VM_FEA).cpu()
                / VM_scaling).squeeze()
                for eps in strain_trace
            ]).cpu()

            diff = strain_trace[:-1] - strain_trace[-1]
            diff = 100 * (diff**2).sum(2).sqrt().mean((-3, -2, -1)).cpu()

        strain_errs[:-1], VM_errs[:-1], diff

        print(strain_errs.shape, VM_errs.shape, diff.shape)

        # take mean and std over instance index (for all iterations)
        se_mean = strain_errs[:-1].mean(dim=1)
        (se_min, _), (se_max, _) =  strain_errs[:-1].min(dim=1),  strain_errs[:-1].max(dim=1)
        ve_mean = VM_errs[:-1].mean(dim=1)
        (ve_min, _), (ve_max, _) =  VM_errs[:-1].min(dim=1),  VM_errs[:-1].max(dim=1)
        diff_mean = diff.mean(dim=1)
        (diff_min, _), (diff_max, _) =  diff.min(dim=1),  diff.max(dim=1)

        # clamp to be non-negative
        # se_min = torch.clamp(se_min, min=0)
        # ve_min = torch.clamp(ve_min, min=0)
        # diff_min = torch.clamp(diff_min, min=0)

        print(se_mean.shape, se_min.shape, se_max.shape)

        xx = np.arange(max_iters-1)

        ax[0].plot(xx, se_mean, "--", c=name_to_color[label], label=label)
        ax[0].fill_between(xx, se_min, se_max, label=label, alpha=0.3, color=name_to_color[label])

        ax[1].plot(xx, ve_mean, "--", c=name_to_color[label], label=label)
        ax[1].fill_between(xx, ve_min, ve_max, label=label, alpha=0.3, color=name_to_color[label])

        ax[2].plot(xx, diff_mean, "--", c=name_to_color[label], label=label)
        ax[2].fill_between(xx, diff_min, diff_max, label=label, alpha=0.3, color=name_to_color[label])

        ax[2].set_yscale('log')

    model_thermino = setup_model(CHECK_THERMINO, CONF_THERMINO)

    eval_model_convergence(model_thermino, "thermino")


    model_deq = setup_model(CHECK_FNODEQ, CONF_FNODEQ)

    eval_model_convergence(model_deq, "fno_deq")


    # model_deq = setup_model(CHECK_IFNO, CONF_IFNO)

    # eval_model_convergence(model_deq, "ifno")


    ax[0].legend()
    ax[0].set_title("Percent RMS Strain Error")
    ax[1].legend()
    ax[1].set_title("Percent L1 VM Stress Error")
    ax[2].legend()
    ax[2].set_title("DEQ Residual")
    ax[2].set_xlabel("Iteration")
    ax[2].xaxis.set_major_locator(tck.MultipleLocator())
    plt.tight_layout()
    plt.legend
    plt.savefig("convergence_comparison.png", dpi=300)


def test_iter_convergence():
    conf_args = load_conf_override(CONF_THERMINO)

    config = Config(**conf_args)

    config.num_voxels = 32
    MAX_ITERS = 16 + 1

    config.add_resid_loss = False
  
    constlaw = StrainToStress_2phase(E_VALS, NU_VALS, E_BAR)
    model = make_localizer(config)
    model.setConstlaw(constlaw)
    load_checkpoint(CHECK_THERMINO, model, strict=True)

    # model = model.cuda()
    # required to get n_states to behave

    print(config)
    print(model)

    m_base = "paper2_32"
    r_base = "paper2_32_u1_responses"
    UPSAMP_MICRO_FAC = 1

    datasets, _ = collect_datasets(m_base, 100.0, r_base=r_base)

    dataset = LocalizationDataset(**datasets[DataMode.VALID])
    m, bc_vals, eps_FEA, sigma_FEA = dataset[PLOT_IND_BAD : PLOT_IND_BAD + 1]


    C_field = model.constlaw.compute_C_field(m)
    # config.num_voxels = 32

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
    print("strain FEA", eps_FEA.mean((-3,-2,-1,0)),eps_FEA.std((-3,-2,-1,0)))
    print("strain pred", last_pred.mean((-3,-2,-1,0)),last_pred.std((-3,-2,-1,0)))

    print("stress FEA", sigma_FEA.mean((-3,-2,-1,0)),eps_FEA.std((-3,-2,-1,0)))
    print("stress pred", model.constlaw(C_field, last_pred).mean((-3,-2,-1,0)),last_pred.std((-3,-2,-1,0)))

    print(last_pred[:, 0].squeeze().shape)
    plot_cube(last_pred[:, 0].squeeze(), "convergence_last.png")
    plot_cube(eps_FEA[:, 0,...].squeeze(), "conv_last_FEA.png")

    print("Computing errors")
    # compute L1 errors as well
    strain_errs = [
        100
        * mean_L1_error(eps[0, 0], eps_FEA[0, 0]).cpu().squeeze()
        / model.constlaw.strain_scaling
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
    ax[0].plot(np.arange(MAX_ITERS-1), strain_errs[:-1])
    ax[0].set_title("Percent Strain Error")
    # ax[0].set_yticks([1, 10, 20, 50],)
    # ax[0].get_yaxis().get_major_formatter().labelOnlyBase = False

    ax[1].plot(np.arange(MAX_ITERS-1), VM_errs[:-1])
    ax[1].set_title("Percent VM Stress Error")
    # ax[1].set_yticks([1, 10, 20, 50],)
    # ax[1].get_yaxis().get_major_formatter().labelOnlyBase = False

    ax[2].semilogy(np.arange(MAX_ITERS - 1), diff)
    ax[2].set_title("DEQ Residual")
    ax[2].set_xlabel("Iteration")
    ax[2].xaxis.set_major_locator(tck.MultipleLocator())
    plt.tight_layout()

    plt.savefig("conv_trace.png", dpi=300)


# test_FFT_multi_2phase()
# test_FFT_iters_2phase()


# test_model_save_load()
test_euler_pred()
test_iter_convergence()
compare_convergences()
# test_super_res()
# test_FFT_iters_crystal()
# prof_C_op()
# test_constlaw()
# test_stiff_ref()
# test_mandel()
# test_rotat()
# test_mat_vec_op()


# test_euler_ang()
# test_fft_deriv()
