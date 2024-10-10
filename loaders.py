import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
import numpy as np
import h5py
import math

from tensor_ops import *
from helpers import upsample_field
from constlaw import *

# increase cache to 1GB
H5_CACHE_SIZE = 4 * 1024 * 1024 * 1024


FAC_STRESS = math.sqrt(2.0)
FAC_STRAIN = math.sqrt(2.0)

E_BAR_DEFAULT = torch.as_tensor([0.001, 0, 0, 0, 0, 0])

E_VALS_DEFAULT = [120.0, 100 * 120.0]
NU_VALS_DEFAULT = [0.3, 0.3]


def to_mandel(vec_ab, fac=1.0, swap_abaqus=False):
    # return vec_ab
    # print(vec_ab.shape, vec_ab[:, 0].shape)
    vec_mand = vec_ab.new_zeros(vec_ab.shape)
    vec_mand[..., 0, :, :, :] = vec_ab[..., 0, :, :, :]  # 11
    vec_mand[..., 1, :, :, :] = vec_ab[..., 1, :, :, :]  # 22
    vec_mand[..., 2, :, :, :] = vec_ab[..., 2, :, :, :]  # 33

    # the factor has two purposes: handle voigt-to-mandel conversion and handle abaqus using engineering shear strain (doubles each term)
    vec_mand[..., 3, :, :, :] = fac * vec_ab[..., 3, :, :, :]  # 23
    vec_mand[..., 4, :, :, :] = fac * vec_ab[..., 4, :, :, :]  # 31
    vec_mand[..., 5, :, :, :] = fac * vec_ab[..., 5, :, :, :]  # 12

    # swap 3rd and 5th entries and convert eng shear strain to shear strain since abaqus is weird
    if swap_abaqus:
        vec_mand[..., [3, 5], :, :, :] = 2.0 * vec_mand[..., [5, 3], :, :, :]

    return vec_mand


class LocalizationDataset(Dataset):
    """Load set of microstructure and strain responses from files"""

    def __init__(
        self,
        micro_file,
        resp_file,
        upsamp_micro_fac=None,
        swap_abaqus=False,
        device=None,
    ):
        # store files
        self.mf = micro_file
        self.rf = resp_file

        self.upsamp_micro_fac = upsamp_micro_fac

        # handles to datasets directly
        self.micro = None
        self.strain = None
        self.stress = None

        self.length = self.getdata(self.mf, dataset_name="micros").shape[0]

        self.constlaw = None

        self.swap_abaqus = swap_abaqus

        # conversion factors for loading into mandel form
        self.fac_strain = FAC_STRAIN
        self.fac_stress = FAC_STRESS

        # phase-wise stiffness matrices
        self.C_mats = None
        self.bc_vals = None
        self.try_phase_info = True
        self.try_bc_vals = True

        # self.device = device

        if swap_abaqus:
            # abaqus uses engineering shear strain, moose does not
            self.fac_strain = math.sqrt(2.0) / 2.0

    def attach_constlaw(self, constlaw):
        # allows preprocessing using constlaw to convert micro -> stiffness
        self.constlaw = constlaw

    def getdata(self, filename, dataset_name=None, opt=None):
        """Actually load data from h5 or numpy file"""
        _, ext = os.path.splitext(filename)

        if ext.lower() in [".h5"]:
            # first open file, then get dataset
            f = h5py.File(filename, "r", rdcc_nbytes=H5_CACHE_SIZE)
            dataset = f.get(dataset_name)

            # # make sure everything is ok
            # if dataset is not None
            return dataset
        elif ext.lower() in [".npy"]:
            f = np.load(filename)
            return f
        else:
            raise NotImplementedError

    def __len__(self):
        """Return the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Get a requested structure-strain pair (or set thereof)"""
        if torch.is_tensor(index):
            index = index.tolist()

        # lazy-load these files to work with multiprocessing
        if self.micro is None:
            self.micro = self.getdata(self.mf, dataset_name="micros")
        if self.strain is None:
            self.strain = self.getdata(self.rf, dataset_name="strain")
        if self.stress is None:
            self.stress = self.getdata(self.rf, dataset_name="stress")

        # try to get phase info and BCs
        if self.C_mats is None and self.try_phase_info:
            self.try_phase_info = False
            phase_info = self.getdata(self.mf, dataset_name="phase_info")
            # if we got any phase info, do some postprocessing now
            if phase_info is not None:
                # now phase info is a dataset of size (n_instances, 2, 2)
                # get unique contrast ratios
                unique_vals = np.unique(phase_info, axis=0)

                # how many different contrast ratios do we have?
                num_unique = len(unique_vals)

                # map contrast ratios to index in big tensor we're going to build
                cr_to_ind = {}

                # list of sets of stiffness tensors
                self.C_mats = torch.zeros(num_unique, 2, 6, 6).float()

                for ind, v in enumerate(unique_vals):
                    # first dump into tuple so it can act as dict key
                    vals = tuple(v.flatten())

                    cr_to_ind[vals] = ind

                    # now build stiffness mat
                    # do 2-phase case for now
                    # separate out physical params
                    E_0, nu_0, E_1, nu_1 = vals

                    # store these stiffness matrices at correct index
                    self.C_mats[ind][0] = isotropic_mandel66(*YMP_to_Lame(E_0, nu_0))
                    self.C_mats[ind][1] = isotropic_mandel66(*YMP_to_Lame(E_1, nu_1))

                # build lookup table mapping each instance index to a stiffness tensor
                self.C_mat_inds = torch.as_tensor(
                    [cr_to_ind[tuple(cr.flatten())] for cr in phase_info]
                )

                # if self.device is not None:
                #     self.C_mats

        if self.bc_vals is None and self.try_bc_vals:
            self.try_bc_vals = False
            self.bc_vals = self.getdata(self.mf, dataset_name="bc_vals")

        # if self.metadata_dict['phase_info'] is not None:
        # if self.meta

        # Load data and get label
        micro = torch.from_numpy(self.micro[index]).float()

        # if we have constlaw, convert directly to stiffness
        if self.constlaw is not None:
            # if we have a constlaw, use it
            micro = self.constlaw.compute_C_field(micro)
        elif self.C_mats is not None:
            # get indices of each instance into tensor of phase_to_C mats
            phase_C_inds = self.C_mat_inds[index]

            # now get phase-to-C map for each instance
            phase_C = self.C_mats[phase_C_inds]

            # print(phase_C.shape, micro.shape)

            # now contract phase dim with C dim to get local C fields
            micro = torch.einsum("...hxyz, ...hrc -> ...rcxyz", micro, phase_C)
        else:
            # just return raw micros if we can't do anything else
            raise NotImplementedError()
            pass

        strain = torch.from_numpy(self.strain[index]).float()
        strain = to_mandel(strain, fac=self.fac_strain, swap_abaqus=self.swap_abaqus)

        stress = torch.from_numpy(self.stress[index]).float()
        stress = to_mandel(stress, fac=self.fac_stress, swap_abaqus=self.swap_abaqus)

        # print(micro.shape, strain.shape)
        # upsample microstructures to match strain shapes
        if self.upsamp_micro_fac is not None:
            micro = upsample_field(micro, self.upsamp_micro_fac)

        # print(micro.shape, strain.shape)
        # make sure spatial dims match
        # print(micro.shape, strain.shape)
        # assert micro.shape[-3:] == strain.shape[-3:]

        # expand along batch size (if one exists)
        # bc_vals = strain.new_ones(
        #     strain.shape[-5:-3] + (1, 1, 1)
        # )

        if self.bc_vals is not None:
            bc_vals = torch.from_numpy(self.bc_vals[index]).float()
        else:
            bc_vals = E_BAR_DEFAULT.expand(strain.shape[-5:-3])

        # either way fix shape
        bc_vals = bc_vals.reshape(strain.shape[-5:-3] + (1, 1, 1))

        # print("strain", strain.shape)
        # print("bc", bc_vals.shape)
        # print("bcv", bc_vals.mean((-3, -2, -1)))
        # print("strain_mean", strain.mean((-3, -2, -1)))
        # print("stress_mean", stress.mean((-3, -2, -1)))

        return micro, bc_vals, strain, stress
