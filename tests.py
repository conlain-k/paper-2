import torch

from constlaw import *
from helpers import *
from tensor_ops import *
from plot_cube import *
from euler_ang import *
from config import Config
from solvers import make_localizer
import time
from h5py import File

import numpy as np

BS = 32
N = 31
k = 2 * PI / N

E_VALS = [120.0, 100 * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

C11, C12, C44 = 160, 70, 60

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

    # print(euler_ang)

    # print(euler_ang.shape)

    checkpoint_file = "checkpoints/deq_best.ckpt"
    conf_file = "configs/fno_deq.json"

    conf_args = load_conf_override(conf_file)
    # print(conf_args)
    config = Config(**conf_args)
    config.num_voxels = 62

    # no need to return residual
    config.return_resid = False

    model = make_localizer(config)

    # override crystalline const law
    model.setConstParams(E_VALS, NU_VALS, E_BAR)
    model.set_constlaw_crystal(C11, C12, C44)

    # pull in saved weights
    load_checkpoint(checkpoint_file, model, strict=False)

    # make sure we're ready to eval
    model = model.cuda()
    model.eval()

    import time

    print("running inference!")

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        # evaluate on batch of euler angles
        strain_preds = model.forward(euler_ang[None])

    torch.cuda.synchronize()
    dt = time.time() - start

    # print(strain_preds.shape)

    print("elapsed time", dt)

    C_field = model.constlaw.compute_C_field(euler_ang[None])

    stress_preds = model.constlaw.forward(strain_preds, C_field)

    plot_cube(
        strain_preds[0, 0, :, :, :].detach().cpu().numpy(),
        "d3d_exx.png",
        cmap="coolwarm",
    )
    plot_cube(
        stress_preds[0, 0, :, :, :].detach().cpu().numpy(),
        "d3d_sxx.png",
        cmap="coolwarm",
    )
    plot_cube(
        C_field[0, 0, 0, :, :, :].detach().cpu().numpy(), "d3d_C11.png", cmap="coolwarm"
    )

    print("\t\tHHHH", C_field[0, 0, 0, :, :, :])
    print(C_field[0, 0, 0, :, :, :].shape)
    plot_cube(
        euler_ang[..., 0].detach().cpu().numpy(),
        "euler_ang.png",
        cmap="coolwarm",
    )

    # dump predictions to a file
    f = File("crystal_pred.h5", "w")
    write_dataset_to_h5(C_field, "C_field", f)
    write_dataset_to_h5(strain_preds, "strain", f)
    write_dataset_to_h5(stress_preds, "stress", f)
    write_dataset_to_h5(euler_ang.unsqueeze(0), "euler_ang", f)


test_stiff_ref()
test_mandel()
test_rotat()
test_euler_pred()


# test_euler_ang()
# prof_C_op()
# test_mat_vec_op()
# test_fft_deriv()
