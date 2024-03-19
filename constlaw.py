import torch
from torch import pi as PI
import itertools

from helpers import *


class StrainToStress_2phase(torch.nn.Module):
    def __init__(self, E_vals, nu_vals):
        # computes stresses from strains and stiffness tensor
        super().__init__()
        # store Lamé coefficients for simplicity
        self.nphases = len(E_vals)
        self.lamb_vals = torch.zeros(self.nphases)
        self.mu_vals = torch.zeros(self.nphases)
        # hold stiffnes matrix for each phase
        self.register_buffer("stiffness_mats", torch.zeros((self.nphases, 6, 6)))
        self.register_buffer("compliance_mats", torch.zeros((self.nphases, 6, 6)))

        for j in range(self.nphases):
            # compute Lamé constants
            self.lamb_vals[j], self.mu_vals[j] = YMP_to_Lame(E_vals[j], nu_vals[j])

            # cache stiffness matrix for later
            self.stiffness_mats[j] = self.compute_C_matrix(
                self.lamb_vals[j], self.mu_vals[j]
            )

            # also cache compliance (inverse stiffness)
            self.compliance_mats[j] = torch.linalg.inv(self.stiffness_mats[j])

        self.lamb_0 = self.lamb_vals.mean()
        self.mu_0 = self.mu_vals.mean()
        # store reference stiffness matrix
        self.register_buffer("C_ref", torch.zeros((1, 6, 6)))
        self.register_buffer("S_ref", torch.zeros((1, 6, 6)))

        # reference-phase stiffness and compliance
        self.C_ref = self.compute_C_matrix(self.lamb_0, self.mu_0)
        self.S_ref = torch.linalg.inv(self.C_ref)

    def compute_C_matrix(self, lamb, mu):
        new_mat = torch.zeros((6, 6), dtype=torch.float32, requires_grad=False)
        # set up tensor
        for row in range(3):
            # set up off-diag in this row
            new_mat[row, :3] = lamb
            # and diag entry
            new_mat[row, row] = 2 * mu + lamb

        for row in range(3, 6):
            # set up last three diagonals
            new_mat[row, row] = mu

        return new_mat

    def forward(self, strain, micro):
        """Apply a given constitutive law over a batch of n-phase microstructures"""
        assert micro is not None
        # y = torch.zeros_like(strain)

        # h is phase num, xyz are space, r,c index C_rc mapping strains strain_c to stresses stress_r
        stress = torch.einsum(
            "bhxyz, hrc, bcxyz->brxyz", micro, self.stiffness_mats, strain
        )

        return stress

    def inverse(self, stress, micro):
        assert micro is not None

        strain = torch.einsum(
            "bhxyz, hrc, bcxyz->brxyz", micro, self.compliance_mats, stress
        )
        return strain

    def stress_pol(self, strain, micro):
        stress = self.forward(strain, micro)
        # compute stress polarization
        stress_polarization = stress - torch.einsum(
            "rc, bcijk -> brijk", self.C_ref, strain
        )

        return stress_polarization


class GreensOp(torch.nn.Module):
    def __init__(self, lamb_0, mu_0, N):
        super().__init__()

        # store Lamé coefficients
        # self.lam, self.mu = YMP_to_Lamé(E, nu)
        self.lamb = lamb_0
        self.mu = mu_0

        # how many voxels in each direction?
        self.N = N

        # precompute Green's op in freq space
        self.register_buffer("G_freq", self._compute_coeffs(self.N))

    def _compute_coeffs(self, N):
        # integer freq terms
        ksi = torch.fft.fftfreq(N) * 2 * PI
        # get it as 3 terms
        ksi_vec = torch.meshgrid(ksi, ksi, ksi, indexing="ij")
        # stack into a frequency vector
        ksi_vec = torch.stack(ksi_vec, dim=0)

        # take element-wise squaring (e.g. squared 2-norm)
        ksi_2 = (ksi_vec**2).sum(dim=0)
        # don't change norm of zero element (since it's already zero)
        ksi_2[..., 0, 0, 0] = 1

        ksi_vec = ksi_vec / ksi_2.sqrt()

        # two coefficients in Green's op calculation
        coef_1 = 1 / 4 * self.mu
        coef_2 = (self.lamb + self.mu) / (self.mu * (self.lamb + 2 * self.mu))

        delta = lambda i, j: int(i == j)

        G = torch.zeros(3, 3, 3, 3, N, N, N, dtype=torch.cfloat)
        # sweep over every possibly combination of indices
        for ind in itertools.product([0, 1, 2], repeat=4):
            i, j, k, h = ind
            # grab elements of ksi vector
            ksi_i, ksi_j, ksi_k, ksi_h = ksi_vec[i], ksi_vec[j], ksi_vec[k], ksi_vec[h]

            # compute each term in the Green's op expression
            term_1 = (
                delta(k, i) * ksi_h * ksi_j
                + delta(h, i) * ksi_k * ksi_j
                + delta(k, j) * ksi_h * ksi_i
                + delta(h, j) * ksi_k * ksi_i
            )
            term_2 = ksi_i * ksi_j * ksi_k * ksi_h

            G[i, j, k, h] = coef_1 * term_1 - coef_2 * term_2

        # Make sure zero-freq terms are actually zero
        # G[..., 0,0,0] = 0
        return G

    def forward(self, tau):
        tau_mat = vec_to_mat(tau)
        # given a stress-like quantity tau, apply the green's tensor via fourier space to get a strain-like quantity eps
        tau_ft = torch.fft.fftn(tau_mat, dim=(-3, -2, -1))
        # do an element-wise matrix multiplication across all frequencies and batch instances
        eps_ft = torch.einsum("ijkhxyz, bkhxyz -> bijxyz", self.G_freq, tau_ft)

        eps_ft_vec = mat_to_vec(eps_ft)

        print(eps_ft_vec.shape)

        # re-set zero-freq term later
        eps_ft_vec[:, :, 0, 0, 0] = 0.0
        eps_ft_vec[:, 0, 0, 0, 0] = 0.001
        print(eps_ft_vec.shape)
        print(eps_ft_vec[:, :, 0, 0, 0])

        eps = torch.fft.ifftn(eps_ft_vec, dim=(-3, -2, -1), s=tau.shape[-3:])

        # now explicitly truncate
        eps = eps.real

        # eps_vec = mat_to_vec(eps)

        # now we have a "strain" field!
        return eps


def VMStress(stress):
    stress_dev_mat = stressdeviator(stress)

    # sum out via double dot product
    inner = torch.einsum("bijxyz, bijxyz -> bxyz", stress_dev_mat, stress_dev_mat)
    # add back channel dim
    inner = inner.unsqueeze(1)

    vm_stress = ((3.0 / 2.0) * inner).sqrt()

    # print(vm_stress.shape)

    # sum over first two dimensions
    return vm_stress


def stressdeviator(stress):
    stress_mat = vec_to_mat(stress)
    trace = stress_mat[:, 0, 0] + stress_mat[:, 1, 1] + stress_mat[:, 2, 2]
    stress_dev_mat = stress_mat

    # subtract off mean stress from diagonal
    stress_dev_mat[:, 0, 0] -= trace / 3
    stress_dev_mat[:, 1, 1] -= trace / 3
    stress_dev_mat[:, 2, 2] -= trace / 3

    return stress_dev_mat


def YMP_to_Lame(E, nu):
    # convert Young's modulus + Poisson Ratio -> Lamé coefficients
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lamb, mu


# given a batch of vector fields R^3 -> R^d, return the gradient for each batch and component
def batched_vector_FFT_grad(a, disc=False):
    # assumes f is [b x d x i x j x k]
    # b: batch index
    # d: channel of output (e.g. displacement component)
    # i, j, k: spatial dimensions
    n = a.shape[-1]
    L = n
    # assumes domain is zero to n
    h = L / n

    # first take fourier transform of signal (for each batch and channel)
    a_FT = torch.fft.fftn(a, dim=(-3, -2, -1))
    # assume all spatial dims are the same
    s = torch.fft.fftfreq(n).to(a_FT.device) * n

    # Get fourier freqs for a given grid
    filt = s * 2 * PI * h * 1j

    if disc:
        # use filter to account for piecewise-constant fields
        filt = torch.sin(2 * PI * s / n) / h
    # x-filter affects x-direction, etc.
    filt_x = filt.reshape(1, 1, -1, 1, 1)
    filt_y = filt.reshape(1, 1, 1, -1, 1)
    filt_z = filt.reshape(1, 1, 1, 1, -1)

    da_dx_FT = a_FT * filt_x * 1j
    da_dy_FT = a_FT * filt_y * 1j
    da_dz_FT = a_FT * filt_z * 1j

    # dimension and size for output
    dd = (-3, -2, -1)
    ss = a.shape[-3:]

    da_dx = torch.fft.ifftn(da_dx_FT, dim=dd, s=ss).real
    da_dy = torch.fft.ifftn(da_dy_FT, dim=dd, s=ss).real
    da_dz = torch.fft.ifftn(da_dz_FT, dim=dd, s=ss).real

    return da_dx, da_dy, da_dz

    # add a dimension for gradient BEFORE channel
    # grad_a_FT = torch.stack([da_dx_FT, da_dy_FT, da_dz_FT], axis=1)
    # grad_a = torch.fft.ifftn(grad_a_FT, dim=(-3, -2, -1), s=a.shape[-3:])
    # return grad_a.real


def central_diff_3d(ff, h=1):
    # takes first-order central difference of field ff
    dx = (torch.roll(ff, -1, dims=-3) - torch.roll(ff, 1, dims=-3)) / (2.0 * h)
    dy = (torch.roll(ff, -1, dims=-2) - torch.roll(ff, 1, dims=-2)) / (2.0 * h)
    dz = (torch.roll(ff, -1, dims=-1) - torch.roll(ff, 1, dims=-1)) / (2.0 * h)

    return [dx, dy, dz]

    # add a dimension for gradient BEFORE channel
    # grad_ff = torch.stack([dx, dy, dz], axis=1)
    # return grad_ff


def est_homog(strain, stress, inds):
    # estimate homogenized stiffness tensor values for given indices
    i, j = inds
    sig_bar = stress[:, i].mean(dim=(-3, -2, -1))
    eps_bar = strain[:, j].mean(dim=(-3, -2, -1))
    # auto-broadcast over batches
    Cij = sig_bar / eps_bar

    return Cij


def stressdiv(stress, use_FFT_deriv=True):

    stress_mat = vec_to_mat(stress)

    # print(stress_mat.shape)
    # compute average stress divergence for a given stress field using either Fourier derivatives or finite differences
    if use_FFT_deriv:
        [dx, dy, dz] = batched_vector_FFT_grad(stress_mat, disc=True)
    else:
        [dx, dy, dz] = central_diff_3d(stress_mat)

    # voxel-wise divergence
    # dsig_1j / dx1 + dsig_2j / dx2 + dsig_3j / dx3
    div = dx[:, 0] + dy[:, 1] + dz[:, 2]

    # now take vector norm for each location
    div = (div**2).sum(dim=1, keepdim=True).sqrt()

    return div


def compute_strain_from_displacment(disp, use_FFT_deriv=False):
    if use_FFT_deriv:
        [dx, dy, dz] = batched_vector_FFT_grad(disp)
    else:
        [dx, dy, dz] = central_diff_3d(disp)
    # now we have b x d1 x d2 x i x j x k (d1 is derivative direction, d2 is component of u)

    # keep batch and spatial dims, 6 channels output
    strain_shape = dx.shape[0:1] + (6,) + dx.shape[-3:]
    strain = dx.new_zeros(strain_shape)

    # ABAQUS order is 11, 22, 33, 12, 13, 23

    # eps_11 = du1/dx
    strain[:, 0] = dx[:, 0]

    # eps_22 = du2/dy
    strain[:, 1] = dy[:, 1]
    # eps_33
    strain[:, 2] = dz[:, 2]

    # eps_12 = du1/dy + du2 / dx
    strain[:, 3] = (dx[:, 1] + dy[:, 0]) / 2
    # eps_13
    strain[:, 4] = (dz[:, 0] + dx[:, 2]) / 2
    # eps_23
    strain[:, 5] = (dy[:, 2] + dz[:, 1]) / 2

    return strain


def compute_strain_energy(strain, stress):
    # first compute elementwise strain ED, then add back a "channel" dimension
    U = torch.einsum("brxyz, brxyz -> bxyz", strain, stress)
    U = U.unsqueeze(1)
    return U
