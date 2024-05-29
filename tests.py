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

BS = 32
N = 31
k = 2 * PI / N

E_VALS = [120.0, 100 * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

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

    euler_ang = torch.from_numpy(euler_ang).cuda()

    checkpoint_file = "checkpoints/fno_deq_best.ckpt"
    conf_file = "configs/fno_deq.json"

    conf_args = load_conf_override(conf_file)
    # conf_args["use_stress_polarization"] = True
    # conf_args["use_energy"] = True
    # print(conf_args)
    config = Config(**conf_args)
    config.num_voxels = 62

    # no need to return residual
    config.return_resid = False

    model = make_localizer(config)

    model.setConstParams(E_VALS, NU_VALS, E_BAR)

    del model.greens_op

    # override to set crystalline const law

    # pull in saved weights
    load_checkpoint(checkpoint_file, model, strict=False)

    model.set_constlaw_crystal(C11, C12, C44)
    model.greens_op = GreensOp(model.constlaw, 62)

    # make sure we're ready to eval
    model = model.cuda()
    model.eval()

    import time

    print("running inference!")

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        # evaluate on batch of euler angles
        strain_pred = model.forward(euler_ang[None])

    torch.cuda.synchronize()
    dt = time.time() - start

    # print(strain_pred.shape)

    print("elapsed time", dt)

    C_field = model.constlaw.compute_C_field(euler_ang[None])

    stress_pred = model.constlaw.forward(strain_pred, C_field)

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
        )
        homog_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()

    print("homog shape", homog_true.shape)

    print(
        f"\nHomog true: {homog_true:5f} pred: {homog_pred:5f} rel_err: {(homog_true - homog_pred)/homog_true:5f}"
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

    strain_11_min = None  # strain_true[0, 0].min()
    strain_11_max = None  # strain_true[0, 0].max()

    stress_11_min = None  # stress_true[0, 0].min()
    stress_11_max = None  # stress_true[0, 0].max()

    plot_cube(
        strain_pred[0, 0, :, :, :].detach().cpu(),
        "poly_extrap_exx.png",
        cmap="coolwarm",
        vmin=strain_11_min,
        vmax=strain_11_max,
    )
    plot_cube(
        stress_pred[0, 0, :, :, :].detach().cpu(),
        "poly_extrap_sxx.png",
        cmap="coolwarm",
        vmin=stress_11_min,
        vmax=stress_11_max,
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

    print("\t\tHHHH", C_field[0, 0, 0, :, :, :])
    print(C_field[0, 0, 0, :, :, :].shape)
    plot_cube(
        euler_ang[..., 0].detach().cpu(),
        "euler_ang.png",
        cmap="coolwarm",
    )

    # dump predictions to a file
    f = File("crystal_pred.h5", "w")
    write_dataset_to_h5(C_field, "C_field", f)
    write_dataset_to_h5(strain_pred, "strain", f)
    write_dataset_to_h5(stress_pred, "stress", f)
    write_dataset_to_h5(euler_ang.unsqueeze(0), "euler_ang", f)


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

    if torch.cuda.is_available():
        C_field = C_field.cuda()
        eps = eps.cuda()
        constlaw = constlaw.cuda()
        G = G.cuda()

    eps_0 = eps
    for i in range(10):
        eps = G.forward(eps, C_field)
        sig = constlaw(eps, C_field)
        equib_err, compat_err = G.compute_residuals(eps, sig)
        if i % 1 == 0:
            print(
                f"Iter {i} equib: {constlaw.C0_norm(equib_err).mean()} compat {constlaw.C0_norm(compat_err).mean()}"
            )
            plot_cube(
                eps[0, 0].detach().cpu(), savedir=f"FFT_eps_{i}.png", cmap="coolwarm"
            )
        # f = File(f"crystal_fft_{i}.h5", "w")
        # write_dataset_to_h5(C_field, "C_field", f)
        # write_dataset_to_h5(eps, "strain", f)

    # print(constlaw.C0_norm(equib_err)[0].shape)
    # print(constlaw.C0_norm(compat_err)[0].shape)


def test_FFT_iters_2phase():

    datasets, _ = collect_datasets("paper2_smooth", 100.0)
    dataset = LocalizationDataset(**datasets[DataMode.TRAIN])
    m, eps_FEA, sigma_FEA = dataset[0:1]

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

    # print(datasets)

    # print(dataset.length)

    # print(dataset[0].shape)

    print(m.shape)

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

    sigma_init = constlaw.forward(eps, C_field)

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
    sigma_rand = constlaw.forward(eps_rand, C_field)

    rand_resid_equi, rand_resid_compat = G.compute_residuals(eps_rand, sigma_rand)
    rand_div_sigma_FT = stressdiv(sigma_rand).mean()
    rand_equi_err = constlaw.C0_norm(rand_resid_equi).mean()
    rand_compat_err = constlaw.C0_norm(rand_resid_compat).mean()

    print(
        f"rand Residuals, div sigma = {rand_div_sigma_FT:4f}, equi err = {rand_equi_err:4f}, compat err = {rand_compat_err:4f}"
    )

    print(f"\t C homog FEA is {C_homog_FEA:4f}")

    MAX_ITERS = 200

    div_sigma_FT = torch.zeros(200)
    equi_err = torch.zeros(200)
    compat_err = torch.zeros(200)
    C_homog = torch.zeros(200)

    for i in range(20):
        eps = G.forward(eps, C_field)
        sigma = constlaw.forward(eps, C_field)

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

    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    x = torch.arange(MAX_ITERS)
    ax[0].plot(x, div_sigma_FT)
    ax[0].set_title("Stress Divergence (FFT)")

    ax[1].plot(x, equi_err)
    ax[1].set_title("Equilibrium Error (C0)")

    ax[2].plot(x, compat_err)
    ax[2].set_title("Compatibility Error (C0)")

    ax[3].plot(x, abs(C_homog - C_homog_FEA))
    ax[3].set_title("C_homog error")

    fig.tight_layout()
    plt.savefig("FFT_convergence_trace.png", dpi=300)


def test_deq_convergence():
    checkpoint_file = "checkpoints/fno_deq_best.ckpt"
    conf_file = "configs/fno_deq.json"

    conf_args = load_conf_override(conf_file)

    config = Config(**conf_args)
    config.return_deq_trace = True
    N_MAX = 32

    config.deq_args["f_max_iter"] = N_MAX
    config.deq_args["n_states"] = N_MAX

    model = make_localizer(config)
    model.setConstParams(E_VALS, NU_VALS, E_BAR)
    load_checkpoint(checkpoint_file, model, strict=False)
    # model = model.cuda()
    # required to get n_states to behave
    model.train()

    print(config)
    print(model)

    datasets, _ = collect_datasets("paper2_smooth", 100.0)
    dataset = LocalizationDataset(**datasets[DataMode.TRAIN])
    m, eps_FEA, sigma_FEA = dataset[0:1]

    print("Evaluating model")
    # don't store any gradients
    with torch.inference_mode():
        strain_trace = model(m)
    print(strain_trace[0].shape)
    strain_trace = torch.stack(strain_trace, dim=0)
    print(strain_trace.shape)

    print("Computing errors")
    # compute L1 errors as well
    errs = [
        mean_L1_error(eps[0, 0], eps_FEA[0, 0]).cpu().numpy()
        / model.constlaw.strain_scaling
        for eps in strain_trace
    ]
    # sum difference along each component
    diff = (
        torch.diff(strain_trace, dim=0)
        .abs()
        .mean(dim=(-3, -2, -1))
        .squeeze()
        .sum(dim=1)
        .cpu()
        .numpy()
    )

    print(diff.shape)
    print(errs[0].shape)

    print("plotting")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.arange(N_MAX), errs)
    ax[0].set_title("StrainError")
    ax[1].semilogy(np.arange(1, N_MAX), diff)
    ax[1].set_title("Update")
    plt.tight_layout()

    plt.savefig("conv_trace.png", dpi=300)


test_euler_pred()
test_deq_convergence()
# test_FFT_iters_2phase()


# test_FFT_iters_crystal()
# test_stiff_ref()
# test_mandel()
# test_rotat()


# test_euler_ang()
# prof_C_op()
# test_mat_vec_op()
# test_fft_deriv()
