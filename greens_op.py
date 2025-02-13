import torch

import itertools
from helpers import *
from tensor_ops import *
from torch import pi as PI

from enum import Enum


# Python 3.9 is weird about StrEnum
class NyquistMethod(str, Enum):
    # set of methods to handle nyquist frequency
    STRAIN = "STRAIN"
    STRESS = "STRESS"
    SYM = "SYM"
    NONE = "NONE"


class GreensOp(torch.nn.Module):
    def __init__(
        self,
        constlaw,
        N,
        nyquist_method=NyquistMethod.STRESS,
        willot=True,
    ):
        super().__init__()

        # store Constitutive law
        self.constlaw = constlaw

        # how many voxels in each direction?
        self.N = N

        # whether to set last freq strains to zero (else set sresses to zero)
        self.nyquist_method = nyquist_method
        if N is not None and N > 0 and self.constlaw is not None:
            # precompute Green's op in freq space
            self.register_buffer(
                "G_freq",
                self._compute_coeffs(self.N, willot=willot),
                persistent=False,
            )

    def get_freqs(self, N, willot):

        L = 1
        h = L / N
        s = 2 * PI / (h * N)

        # integer freq terms, assuming unit domain length
        q = torch.fft.fftfreq(N, d=h)
        # use convention from Willot and Brisard that nyquist freq is positive
        if N % 2 == 0:
            q[N // 2] *= -1

        # print(q)

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

    def get_G_entry(self, q):
        coef_1 = 1.0 / (4.0 * self.constlaw.mu_0)
        coef_2 = (self.constlaw.lamb_0 + self.constlaw.mu_0) / (
            self.constlaw.mu_0 * (self.constlaw.lamb_0 + 2.0 * self.constlaw.mu_0)
        )
        # get entries in G for a single or batch of frequencies q
        ret = torch.zeros((3, 3, 3, 3) + q.shape[1:], dtype=torch.cfloat)
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

            ret[i, j, k, h, ...] = coef_1 * term_1 - coef_2 * term_2
        return ret

    def _compute_coeffs(self, N, willot):
        q = self.get_freqs(N, willot=willot)

        # print(q.shape)
        normq = (q**2).sum(dim=0, keepdim=True).sqrt()
        # fix norm term to not divide by zero
        normq[:, 0, 0, 0] = 1

        q /= normq

        # print("qq")
        # print(q[:, 2, 2, 16])
        # print(q[:, 30, 30, 16])

        G = self.get_G_entry(q)

        # if we have even voxel count and are told to do some filteriong
        if N % 2 == 0 and self.nyquist_method != NyquistMethod.NONE:

            # print("fixing nyquist freqs")
            N_half = N // 2
            s1 = np.s_[..., N_half, :, :]
            s2 = np.s_[..., :, N_half, :]
            s3 = np.s_[..., :, :, N_half]

            if self.nyquist_method == NyquistMethod.SYM:
                # note quite working yet
                raise NotImplementedError()
                G_copy = G.clone().detach()
                # iterate over nyquist freqs and force symmetry
                for i in range(1, self.N // 2):
                    for j in range(1, self.N // 2):
                        for k in range(1, self.N // 2):
                            print(i, j, k)
                            li = 1 + i
                            hi = self.N - i - 1
                            lj = 1 + j
                            hj = self.N - j - 1
                            lk = 1 + k
                            hk = self.N - k - 1

                            G_copy[..., N_half, lj, lk] = (
                                G[..., N_half, lj, lk] + G[..., N_half, hj, hk].conj()
                            ) / 2.0
                            G_copy[..., N_half, hj, hk] = (
                                G[..., N_half, hj, hk] + G[..., N_half, lj, lk].conj()
                            ) / 2.0

                            G_copy[..., li, N_half, lk] = (
                                G[..., li, N_half, lk] + G[..., hi, N_half, hk].conj()
                            ) / 2.0
                            G_copy[..., hi, N_half, hk] = (
                                G[..., hi, N_half, hk] + G[..., li, N_half, lk].conj()
                            ) / 2.0

                            G_copy[..., li, lj, N_half] = (
                                G[..., li, lj, N_half] + G[..., hi, hj, N_half].conj()
                            ) / 2.0
                            G_copy[..., hi, hj, N_half] = (
                                G[..., hi, hj, N_half] + G[..., li, lj, N_half].conj()
                            ) / 2.0

                G = G_copy

            elif self.nyquist_method == NyquistMethod.STRAIN:
                # print("nyquist strain zero")
                G[s1] = 0
                G[s2] = 0
                G[s3] = 0

            else:
                # print("nyquist stress zero")
                s_ref = C_mandel_to_mat_3x3x3x3(self.constlaw.S_ref)[..., None, None]
                s_ref = s_ref.expand(G[..., N_half, :, :].shape)

                # if even, zero out all Nyquist freqs as well (either strain or stress vals)
                G[s1] = s_ref
                G[s2] = s_ref
                G[s3] = s_ref

        # Make sure zero-freq terms are actually zero for each component
        G[..., 0, 0, 0] = 0

        G_unsym = G

        # convert to symmetric 6x6 representation
        G_sym = C_3x3x3x3_to_mandel(G.unsqueeze(0)).squeeze()

        G_sym_back = C_mandel_to_mat_3x3x3x3(G_sym.unsqueeze(0)).squeeze()

        # make sure we actually compute a symmetric tensor
        assert torch.allclose(G_unsym, G_sym_back)

        return G_sym

    def forward(self, eps_k, C_field, use_polar=False):
        # TODO more involved frequency checking
        assert self.N > 0
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
            sigma = self.constlaw(C_field, eps_k)
            eps_pert = self.apply_gamma(sigma)
            # eps_pert = eps_pert - eps_pert.mean(dim=(-3, -2, -1), keepdim=True)
            eps_kp = eps_k - eps_pert

        # print(eps_kp.mean((-3, -2, -1)))
        return eps_kp

    def apply_gamma(self, x):
        # apply green's op directly to a given symmetric tensor field (via FFT)
        # assumes size is b,6,x,y,z

        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        # print(self.G_freq.shape, x_ft.shape)
        y_ft = torch.einsum("ijxyz, bjxyz -> bixyz", self.G_freq, x_ft)

        # drop back to vector format
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

        return self.apply_gamma(C0e)

    def project_S(self, field):
        # remove orthogonal projection
        return field - self.project_D(field)

    def compute_residuals(self, strain, stress):
        # compute equilibrium and compatibility residuals in L-S sense
        resid_equi = torch.einsum("rc, bcijk -> brijk", self.constlaw.S_ref, stress)
        resid_equi = self.project_D(resid_equi)

        # normalize by avg strain (assumes average matches true avg and is nonzero)
        mean_strain = strain.mean(dim=(-3, -2, -1), keepdim=True)
        # take L2 norm of mean strain
        mean_strain_scale = (mean_strain**2).sum(dim=1, keepdim=True).sqrt()

        # also remove mean strain from residual term
        resid_compat = self.project_S(strain - mean_strain)

        # avoid ever dividing by zero, but broadcast along components/batch
        if mean_strain_scale.all() > 0:
            resid_equi /= mean_strain_scale
            resid_compat /= mean_strain_scale

        return resid_equi, resid_compat
