import torch

import itertools
from helpers import *
from tensor_ops import *

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
            "G_freq", self._compute_coeffs(self.N, filt=False, willot=True)
        )

        freqs_base = self.get_freqs(N, willot=False)
        freqs_willot = self.get_freqs(N, willot=True)

        # print("freqs_base", freqs_base[0, 1, 0])
        # print("freqs_willot", freqs_willot[0, 1, 0])

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

        L = 1
        h = L / N
        s = 2 * PI / (h * N)

        # integer freq terms, assuming unit domain length
        q = torch.fft.fftfreq(N, d=h)

        # print("q is", q)
        # get it as 3 terms
        q_v = torch.meshgrid(q, q, q, indexing="ij")
        # stack into a frequency vector
        q_vec = torch.stack(q_v, dim=0)
        # print("shape", q.shape, q_vec.shape)

        if willot:
            # print("Using willot terms!")

            # compute phi term from Janus implementation
            # https://github.com/sbrisard/janus/blob/a6196a025fee6bf0f3eb5e636a6b2f895ca6fbc9/janus/green.pyx#L999
            phi = 0.5 * s * h * q_vec

            # TODO optimize (this formula follows Janus to avoid transcription errors)

            c = torch.cos(phi)
            s = torch.sin(phi)

            q_vec[0] = s[0] * c[1] * c[2]
            q_vec[1] = c[0] * s[1] * c[2]
            q_vec[2] = c[0] * c[1] * s[2]

        return q_vec

    # def _compute_coeffs_alt(self, N):
    #     q = self.get_freqs(N)
    #     G = torch.zeros(3, 3, 3, 3, N, N, N, dtype=torch.cfloat)
    #     # sweep over every possibly combination of indices
    #     for ind in itertools.product([0, 1, 2], repeat=4):
    #         i, j, k, h = ind
    #         G[k, h, i, j] = self._G(q, ind)

    def _compute_coeffs(self, N, filt=False, willot=False):

        # two coefficients in Green's op calculation
        coef_1 = 1 / (4 * self.constlaw.mu_0)
        coef_2 = (self.constlaw.lamb_0 + self.constlaw.mu_0) / (
            self.constlaw.mu_0 * (self.constlaw.lamb_0 + 2 * self.constlaw.mu_0)
        )

        q = self.get_freqs(N, willot=willot)
        normq = (q**2).sum(dim=0, keepdim=True).sqrt()
        # fix norm term to not divide by zero
        normq[:, 0, 0, 0] = 1

        q /= normq

        G = torch.zeros(3, 3, 3, 3, N, N, N, dtype=torch.cfloat)
        # sweep over every possibly combination of indices
        for ind in itertools.product([0, 1, 2], repeat=4):
            i, j, k, h = ind
            # grab elements of ksi vector
            q_i = q[i]
            q_j = q[j]
            q_k = q[k]
            q_h = q[h]

            # compute each term in the Green's op expression
            term_1 = (
                delta(k, i) * q_h * q_j
                + delta(h, i) * q_k * q_j
                + delta(k, j) * q_h * q_i
                + delta(h, j) * q_k * q_i
            )
            term_2 = q_i * q_j * q_k * q_h

            G[k, h, i, j] = coef_1 * term_1 - coef_2 * term_2
            # print(i, j, k, h, G[i, j, k, h].abs().sum())

        if filt:
            filter = (1 + torch.cos(PI * normq)) / 2
            G = G * filter.reshape(1, 1, 1, 1, N, N, N)

        # Make sure zero-freq terms are actually zero for each component
        G[..., 0, 0, 0] = 0
        return G

    def forward(self, eps_k, C_field, use_polar=True):
        # given current strain, compute a new one
        if use_polar:

            tau = self.constlaw.stress_pol(
                eps_k,
                C_field,
            )

            # print(C_field.mean(dim=(-3, -2, -1)))
            # print(self.constlaw.C_ref_unscaled)
            # print(self.constlaw.C_ref)
            E_bar = eps_k.mean(dim=(-3, -2, -1), keepdim=True)
            eps_kp = E_bar - self.apply_gamma(tau)

        else:
            sigma = self.constlaw.forward(eps_k, C_field)
            eps_pert = self.apply_gamma(sigma)
            # eps_pert = eps_pert - eps_pert.mean(dim=(-3, -2, -1), keepdim=True)
            eps_kp = eps_k - eps_pert

        return eps_kp

    def apply_gamma(self, x):
        # apply green's op directly to a given symmetric tensor field (via FFT)
        # assumes size is b,6,x,y,z

        # lift into 3x3 matrix
        x = mandel_to_mat_3x3(x)
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        y_ft = torch.einsum("ijkhxyz, bkhxyz -> bijxyz", self.G_freq, x_ft)

        # drop back to vector format
        y_ft = mat_3x3_to_mandel(y_ft)
        y = torch.fft.ifftn(y_ft, dim=(-3, -2, -1), s=x.shape[-3:])

        return y.real

    def project_D(self, field):
        """
        Given a batch of 3D symmetric tensor fields (e.g. stress, strain) project onto the D-subspace (compatible)
        Used to check convergence / as an error metric
        Assumes field has indices b, c, x, y, z and size (batch_size, 6, N, N, N) for N voxels in each direction
        """
        # get C0 : e term for field e
        C0e = torch.einsum("rc, bcxyz -> brxyz", self.constlaw.C_ref, field)

        # print(field.shape, self.constlaw.C_ref.shape)

        return self.apply_gamma(C0e)

    def project_S(self, field):
        # remove orthogonal projection
        return field - self.project_D(field)

    def compute_residuals(self, strain, stress):
        # compute equilibrium and compatibility residuals in L-S sense
        resid_equi = torch.einsum("rc, bcijk -> brijk", self.constlaw.S_ref, stress)
        resid_equi = self.project_D(resid_equi)

        # normali   ze by avg strain (assumes average matches true avg and is nonzero)
        mean_strain = strain.mean(dim=(-3, -2, -1), keepdim=True)
        # take L2 norm of mean strain
        mean_strain_scale = (mean_strain**2).sum(dim=1, keepdim=True).sqrt()

        # print(mean_strain.shape, mean_strain_scale.shape, strain.shape)

        # print(mean_strain.shape)

        # print(mean_strain)

        # also remove mean strain from residual term
        resid_compat = self.project_S(strain - mean_strain)

        # avoid ever dividing by zero, but broadcast along components/batch
        if mean_strain_scale.all() > 0:
            resid_equi /= mean_strain_scale
            resid_compat /= mean_strain_scale

        return resid_equi, resid_compat
