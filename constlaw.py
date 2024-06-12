import torch
from torch import pi as PI
import itertools

from helpers import *
from tensor_ops import *
from einops import rearrange
from euler_ang import *


class StrainToStress_base(torch.nn.Module):
    def __init__(self):
        # computes stresses from strains and stiffness tensor (base class)
        super().__init__()

        # store reference stiffness matrix (need to set values in child class)
        self.register_buffer("C_ref", torch.zeros((6, 6)))
        self.register_buffer("S_ref", torch.zeros((6, 6)))

        self.stiffness_scaling = None
        self.strain_scaling = None
        self.stress_scaling = None
        self.energy_scaling = None

    def compute_scalings(self, eps_bar):
        frob = lambda x: (x**2).sum().sqrt()
        self.stiffness_scaling = frob(self.C_ref)
        self.strain_scaling = frob(eps_bar)

        # compute other scalings downstream of this one
        self.stress_scaling = self.stiffness_scaling * self.strain_scaling
        self.energy_scaling = self.stress_scaling * self.strain_scaling

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

    def stress_pol(self, strain, C_field, scaled=False):
        """
        Compute stress polarization (pass scaled=True if strains and C are scaled)
        """

        C_ref = self.C_ref
        if scaled:
            C_ref = C_ref / self.stiffness_scaling

        C_pert = C_field - C_ref.reshape(1, 6, 6, 1, 1, 1)
        # compute stress polarization
        stress_polarization = torch.einsum("brcxyz, bcxyz -> brxyz", C_pert, strain)

        return stress_polarization

    def forward(self, C_field, strain):
        """Apply a given constitutive law over a batch of n-phase microstructures"""

        stress = torch.einsum("brcxyz, bcxyz->brxyz", C_field, strain)

        return stress

    def C0_norm(self, field, average=False, scaled=False):
        """
        Given a batch of 3D symmetric tensor fields (e.g. strain) compute the C0-norm
        Used to check convergence / as an error metric
        Assumes field has indices b, i, x, y, z and size (batch_size, 6, N, N, N) for N voxels in each direction
        """
        scale = 1.0
        if scaled:
            scale = 1.0 / self.stiffness_scaling
        return weighted_norm(field, self.C_ref * scale, average)

    def S0_norm(self, field, average=False, scaled=False):
        # like the C0-norm, but for stress-like quantities
        scale = 1.0
        if scaled:
            scale = self.stiffness_scaling
        return weighted_norm(field, self.S_ref * scale, average)


class StrainToStress_2phase(StrainToStress_base):
    def __init__(self, E_vals, nu_vals):
        # computes stresses from strains and stiffness tensor (2 phase specialization)
        super().__init__()
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
            self.stiffness_mats[j] = isotropic_mandel66(
                self.lamb_vals[j], self.mu_vals[j]
            )

            # also cache compliance (inverse stiffness)
            self.compliance_mats[j] = torch.linalg.inv(self.stiffness_mats[j])

        # store Lamé coefficients for simplicity

        self.lamb_0 = self.lamb_vals.mean()
        self.mu_0 = self.mu_vals.mean()
        self.C_ref = isotropic_mandel66(self.lamb_0, self.mu_0)
        self.S_ref = torch.linalg.inv(self.C_ref)

    def compute_C_field(self, micros):
        # compute stiffness tensor field from 2-phase microstructure
        return torch.einsum("bhxyz, hrc -> brcxyz", micros, self.stiffness_mats)


class StrainToStress_crystal(StrainToStress_base):
    def __init__(self, C11, C12, C44):
        # computes stresses from strains and stiffness tensor (cubic crystal)
        super().__init__()

        # store cubic mats
        C_unrot = cubic_mandel66(C11, C12, C44)
        C_unrot_3333 = C_mandel_to_mat_3x3x3x3(C_unrot)
        # use aniso reference for now (NOT GOOD)
        self.register_buffer("C_unrot", C_unrot)
        self.register_buffer("C_unrot_3333", C_unrot_3333)

        # project onto "nearest" isotropic tensor
        # uses Norris' formulas https://msp.org/jomms/2006/1-2/jomms-v1-n2-p02-s.pdf

        k = (C11 + 2 * C12) / 3.0
        # cache these for the Green's op as well
        self.mu_0 = (C44**3 * (C11 - C12) ** 2) ** (1.0 / 5.0)
        self.lamb_0 = k - 2.0 * self.mu_0 / 3.0

        # use (nearest) isotropic reference
        self.C_ref = isotropic_mandel66(self.lamb_0, self.mu_0)
        self.S_ref = torch.linalg.inv(self.C_ref)

    def compute_local_stiffness(self, euler_ang, stiff_mat_base):
        # assumes euler ange have shape (batch, 3, x, y, z)

        orig_shape = euler_ang.shape

        # assumes batch is first index, then space, then angle (dream3d order)
        euler_ang = rearrange(euler_ang, "b x y z theta -> (b x y z) theta", theta=3)

        # output is (bxyz, 6, 6)
        C_field = batched_rotate(euler_ang, stiff_mat_base, passive=False)

        C_field = C_3x3x3x3_to_mandel(C_field)

        # unbatch and move channels last
        C_field = rearrange(
            C_field,
            "(b x y z) r c -> b r c x y z",
            b=orig_shape[0],
            x=orig_shape[1],
            y=orig_shape[2],
            z=orig_shape[3],
            r=6,
            c=6,
        )
        # flatten euler angles first

        # print(C_field.shape)
        return C_field

    def compute_C_field(self, micros):
        return self.compute_local_stiffness(micros, self.C_unrot_3333)


def VMStress(stress):
    stress_dev_mat = stressdeviator(stress)

    # sum out via double dot product
    inner = torch.einsum("bijxyz, bijxyz -> bxyz", stress_dev_mat, stress_dev_mat)
    # add back channel dim
    inner = inner.unsqueeze(1)

    vm_stress = ((3.0 / 2.0) * inner).sqrt()

    # sum over first two dimensions
    return vm_stress


def stressdeviator(stress):
    stress_mat = mandel_to_mat_3x3(stress)
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

    stress_mat = mandel_to_mat_3x3(stress)

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
    # add ellipsis to allow for strain-gradient-type calculations
    U = torch.einsum("brxyz, brxyz -> bxyz", strain, stress)
    U = U.unsqueeze(1)
    return U


def flatten_stiffness(C_field):
    # keep batch and space dimensions
    new_shape = (C_field.shape[0], 21) + C_field.shape[-3:]
    # extract 21 unique coeffs
    C_vec = C_field.new_zeros(new_shape)
    # first 6 coeffs take top row
    C_vec[:, 0:6] = C_field[:, 0, :]
    # second row, but skip first
    C_vec[:, 6:11] = C_field[:, 1, 1:]
    # third row, skip first 2
    C_vec[:, 11:15] = C_field[:, 2, 2:]
    # and so on
    C_vec[:, 15:18] = C_field[:, 3, 3:]
    C_vec[:, 18:20] = C_field[:, 4, 4:]
    C_vec[:, 20:21] = C_field[:, 5, 5:]

    return C_vec


def weighted_norm(field, weight_mat, average):
    # contract components (but not space)
    res = torch.einsum("brxyz, rc, bcxyz -> bxyz", field, weight_mat, field)
    # average out over voxels if desired
    if average:
        res = res.mean(dim=(-3, -2, -1))

    return res


def compute_quants(model, strain, C_field):
    # handy helper to compute multiple thermodynamic quantities at once
    stress = model.constlaw(C_field, strain)
    stress_polar = model.constlaw.stress_pol(strain, C_field)
    energy = compute_strain_energy(strain, stress)

    return stress, stress_polar, energy
