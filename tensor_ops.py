import torch
import itertools

import math

from euler_ang import euler_angles_to_matrix

SQRT2 = math.sqrt(2.0)

# where to look in matrix for each vector entry
mandel_ind = torch.as_tensor([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
# where to look in vector for each matrix entry
mandel_ind_inv = torch.as_tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def mat_3x3_to_mandel(mat):
    # requires mat to have shape (batch, 3,3, ...), where the last bit can be anything
    new_shape = mat.shape[0:1] + (6,) + mat.shape[3:]
    vec = mat.new_zeros(new_shape)

    # extract from diagonals
    vec[:, 0] = mat[:, 0, 0]
    vec[:, 1] = mat[:, 1, 1]
    vec[:, 2] = mat[:, 2, 2]

    # off-diag
    vec[:, 3] = mat[:, 0, 1] * SQRT2
    vec[:, 4] = mat[:, 0, 2] * SQRT2
    vec[:, 5] = mat[:, 1, 2] * SQRT2

    return vec


def mandel_to_mat_3x3(vec):
    # requires vec to have shape (batch, 6, ...), where the last bit can be anything
    new_shape = vec.shape[0:1] + (3, 3) + vec.shape[2:]
    mat = vec.new_zeros(new_shape)

    # diagonals
    mat[:, 0, 0] = vec[:, 0]
    mat[:, 1, 1] = vec[:, 1]
    mat[:, 2, 2] = vec[:, 2]

    # off-diag (rescaled)
    mat[:, 0, 1] = vec[:, 3] / SQRT2
    mat[:, 1, 0] = vec[:, 3] / SQRT2
    mat[:, 0, 2] = vec[:, 4] / SQRT2
    mat[:, 2, 0] = vec[:, 4] / SQRT2
    mat[:, 1, 2] = vec[:, 5] / SQRT2
    mat[:, 2, 1] = vec[:, 5] / SQRT2

    return mat


def mandel_C_fac(i, j):
    ret = 1
    if i >= 3:
        ret *= SQRT2
    if j >= 3:
        ret *= SQRT2
    return ret


def C_3x3x3x3_to_mandel(C):
    squeeze_at_end = False
    if len(C.shape) == 4:
        squeeze_at_end = True
        C = C.reshape(1, 3, 3, 3, 3)
    # requires C to have shape (batch, 3,3,3,3, ...), where the last bit can be anything
    new_shape = C.shape[0:1] + (6, 6) + C.shape[5:]
    mat_66 = C.new_zeros(new_shape)

    # loop over target indices
    for i, j in itertools.product(torch.arange(6), repeat=2):
        m, n = mandel_ind[i]
        o, p = mandel_ind[j]

        # now assign value and multiply by mandel scaling
        mat_66[:, i, j] = C[:, m, n, o, p] * mandel_C_fac(i, j)

    if squeeze_at_end:
        mat_66 = mat_66.squeeze()
    return mat_66


def C_mandel_to_mat_3x3x3x3(mat_66):
    squeeze_at_end = False
    # special case if we get non-batched
    if len(mat_66.shape) == 2:
        squeeze_at_end = True
        mat_66 = mat_66.reshape(1, 6, 6)

    # requires mat to have shape (batch, 6,6, ...), where the last bit can be anything
    new_shape = mat_66.shape[0:1] + (3, 3, 3, 3) + mat_66.shape[5:]

    C = mat_66.new_zeros(new_shape)

    # loop over target indices
    for m, n, o, p in itertools.product(torch.arange(3), repeat=4):

        i = mandel_ind_inv[m, n]
        j = mandel_ind_inv[o, p]

        # now assign value and divide by mandel scaling
        C[:, m, n, o, p] = mat_66[:, i, j] / mandel_C_fac(i, j)

    if squeeze_at_end:
        C = C.squeeze()

    return C


def delta(i, j):
    # kronecker delta
    return int(i == j)


def YMP_to_Lame(E, nu):
    # convert Young's modulus + Poisson Ratio -> Lamé coefficients
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lamb, mu


def identity_66():
    return torch.eye(6, 6)


def identity_3333():
    return C_mandel_to_mat_3x3x3x3(identity_66())


def isotropic_mandel66(lamb, mu):
    # extract coefficients and use the fact that isotropic is a subset of cubic
    return cubic_mandel66(2 * mu + lamb, lamb, mu)


def cubic_mandel66(C11, C12, C44):
    # build 6x6 stiffness matrix
    new_mat = torch.zeros((6, 6), dtype=torch.float32, requires_grad=False)

    for row in range(3):
        # set up off-diag in this row
        new_mat[row, :3] = C12
        # and diag entry
        new_mat[row, row] = C11

    for row in range(3, 6):
        # set up last three diagonals
        # mandel fac balances sqrt2 in strain and stress
        new_mat[row, row] = C44 * 2

    return new_mat


# assumes passive rotations by default (Bunge angles)
# if not passive, we need to transpose (flip) the rotation
def batched_rotate(euler_ang, stiff_mat_base, passive=True):
    # assumes euler angles has some characteristic batch and channel size
    # input size is (c, 3) for euler and (3,3,3,3) for base

    # get vector of rotation mats
    R = euler_angles_to_matrix(euler_ang, "ZXZ")

    

    # if passive:
    # flip 1st and 2nd indices (convert passive -> active) to match MOOSE
    # R = R.transpose(1, 2)

    # do operation over flattened spatial vector
    C_field = torch.einsum(
        "m n o p, c i m, c j n, c k o, c l p  -> c i j k l", stiff_mat_base, R, R, R, R
    )

    # now conver
    # C_field = C_3x3x3x3_to_mandel(C_field)

    # should return (c, 6, 6) matrix
    return C_field
