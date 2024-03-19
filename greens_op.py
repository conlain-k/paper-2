import torch

import itertools
from helpers import *

from torch import pi as PI

class GreensOp(torch.nn.Module):
    def __init__(self, constlaw, N):
        super().__init__()

        # store Constitutive law
        self.constlaw = constlaw

        # how many voxels in each direction?
        self.N = N

        # precompute Green's op in freq space
        self.register_buffer(
            "G_freq", self._compute_coeffs(self.N, filt=True, willot=False)
        )

        freqs_base = self.get_freqs(N, willot=False)
        freqs_willot = self.get_freqs(N, willot=True)

        print("freqs_base", freqs_base[0, 1, 0])
        print("freqs_willot", freqs_willot[0, 1, 0])

        # torch.testing.assert_close(
        #     self._compute_coeffs(self.N), self._compute_coeffs_alt(self.N)
        # )

    # def _G(self, q, ind):
    #     # compute continuous Green's op at a frequency k

    #     def N(i, k):
    #         print(self.constlaw.C_ref.shape, q.shape)
    #         K = torch.einsum("ijkh, hxyz, jxyz -> ikxyz", self.constlaw.C_ref, q, q)
    #         return K.inv()

    #     (i, j, k, h) = ind

    #     term1 = N(h, i) * q[j] * q[k]
    #     term2 = N(k, i) * q(j) * q(h)
    #     term3 = N(h, j) * q(i) * q(k)
    #     term4 = N(k, j) * q(i) * q(h)

    #     return (term1 + term2 + term3 + term4) / 4.0

    def get_freqs(self, N, willot):

        # integer freq terms
        q = torch.fft.fftfreq(N, d=1)  # / N  # * 2 * PI / N

        print("q is", q)
        # get it as 3 terms
        q_v = torch.meshgrid(q, q, q, indexing="ij")
        # stack into a frequency vector
        q_vec = torch.stack(q_v, dim=0)
        print("shape", q.shape, q_vec.shape)

        if willot:
            print("Using willot terms!")
            # use the willot modification
            qnorm = (q_vec**2).sum(dim=0, keepdim=True).sqrt()
            qnorm[:, 0, 0, 0] = 1
            n = q_vec / qnorm
            nn = PI * n / N
            # currently domain is zero to N, spanned by N voxels
            hh = N / N
            q_vec = torch.tan(nn) / hh
            # q_vec = (torch.exp(1j * 2 * PI * n / N) - 1) / h

            # q_vec = (
            #     (1j / 4)
            #     * torch.tan(nn / 2)
            #     * (1 + torch.exp(1j * nn[0:1]))
            #     * (1 + torch.exp(1j * nn[1:2]))
            #     * (1 + torch.exp(1j * nn[2:3]))
            # )

            # q_vec = 1j * torch.sin(nn)
            # q_vec =
        return q_vec

    # def _compute_coeffs_alt(self, N):
    #     q = self.get_freqs(N)
    #     G = torch.zeros(3, 3, 3, 3, N, N, N, dtype=torch.cfloat)
    #     # sweep over every possibly combination of indices
    #     for ind in itertools.product([0, 1, 2], repeat=4):
    #         i, j, k, h = ind
    #         G[k, h, i, j] = self._G(q, ind)

    def _compute_coeffs(self, N, filt=False, willot=False):

        q = self.get_freqs(N, willot=willot)
        # two coefficients in Green's op calculation
        coef_1 = 1 / (4 * self.constlaw.mu_0)
        coef_2 = (self.constlaw.lamb_0 + self.constlaw.mu_0) / (
            self.constlaw.mu_0 * (self.constlaw.lamb_0 + 2 * self.constlaw.mu_0)
        )

        normq = (q**2).sum(dim=0, keepdim=True).sqrt()
        # fix norm term to not divide by zero
        normq[:, 0, 0, 0] = 1

        q /= normq

        G = torch.zeros(3, 3, 3, 3, N, N, N, dtype=torch.cfloat)
        # sweep over every possibly combination of indices
        for ind in itertools.product([0, 1, 2], repeat=4):
            i, j, k, h = ind
            # grab elements of ksi vector
            ksi_i = q[i]
            ksi_j = q[j]
            ksi_k = q[k]
            ksi_h = q[h]

            # compute each term in the Green's op expression
            term_1 = (
                delta(k, i) * ksi_h * ksi_j
                + delta(h, i) * ksi_k * ksi_j
                + delta(k, j) * ksi_h * ksi_i
                + delta(h, j) * ksi_k * ksi_i
            )
            term_2 = ksi_i * ksi_j * ksi_k * ksi_h

            G[k, h, i, j] = coef_1 * term_1 - coef_2 * term_2
            # print(i, j, k, h, G[i, j, k, h].abs().sum())

        if filt:
            filter = (1 + torch.cos(PI * normq)) / 2
            G = G * filter.reshape(1, 1, 1, 1, N, N, N)

        # Make sure zero-freq terms are actually zero
        G[..., 0, 0, 0] = 0
        return G

    def forward(self, eps_k, micro, use_polar=False):
        # given current strain, compute a new one
        if use_polar:
            # compute stress polarization
            tau = self.constlaw.stress_pol(eps_k, micro)
            # apply but keep mean
            eps_kp = -self.apply_gamma(tau)
            eps_kp = (
                eps_kp
                - eps_kp.mean(dim=(-3, -2, -1), keepdim=True)
                + eps_k.mean(dim=(-3, -2, -1), keepdim=True)
            )
        else:
            sigma = self.constlaw.forward(eps_k, micro)
            eps_pert = self.apply_gamma(sigma)
            # eps_pert = eps_pert - eps_pert.mean(dim=(-3, -2, -1), keepdim=True)
            eps_kp = eps_k - eps_pert

        return eps_kp

    def apply_gamma(self, x):
        # apply green's op directly to a given symmetric tensor field (via FFT)
        # assumes size is b,6,x,y,z

        # lift into 3x3 matrix
        x = vec_to_mat(x)
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        y_ft = torch.einsum("ijkhxyz, bkhxyz -> bijxyz", self.G_freq, x_ft)

        # drop back to vector format
        y_ft = mat_to_vec(y_ft)
        y = torch.fft.ifftn(y_ft, dim=(-3, -2, -1), s=x.shape[-3:])

        return y.real