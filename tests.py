import torch

from constlaw import *
from helpers import *

BS = 32
N = 31
k = 2 * PI / N


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
    mat = vec_to_mat(vec)
    vec2 = mat_to_vec(mat)
    mat2 = vec_to_mat(vec)

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

    print(X.shape)

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

    plot_pred(-1, torch.zeros(N, N), gfx[3, 0, :, :, 0], dx[3, 0, :, :, 0], "dx")

    torch.testing.assert_close(gfx, dx)
    torch.testing.assert_close(gfy, dy)
    torch.testing.assert_close(gfz, dz)


test_mat_vec_op()
test_fft_deriv()
