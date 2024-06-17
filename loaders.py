import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
import numpy as np
import h5py
import math

from tensor_ops import *

# increase cache to 1GB
H5_CACHE_SIZE = 4 * 1024 * 1024 * 1024

FAC_STRESS = math.sqrt(2.0)
FAC_STRAIN = math.sqrt(2.0) / 2.0

E_BAR_DEFAULT = torch.as_tensor([0.001, 0, 0, 0, 0, 0])

E_VALS_DEFAULT = [120.0, 100 * 120.0]
NU_VALS_DEFAULT = [0.3, 0.3]


def abaqus_to_mandel(vec_ab, fac=1.0):
    # return vec_ab
    # print(vec_ab.shape, vec_ab[:, 0].shape)
    vec_mand = vec_ab.new_zeros(vec_ab.shape)
    vec_mand[..., 0, :, :, :] = vec_ab[..., 0, :, :, :]  # 11
    vec_mand[..., 1, :, :, :] = vec_ab[..., 1, :, :, :]  # 22
    vec_mand[..., 2, :, :, :] = vec_ab[..., 2, :, :, :]  # 33

    # the factor has two purposes: handle voigt-to-mandel conversion and handle abaqus using engineering shear strain (doubles each term)
    vec_mand[..., 3, :, :, :] = fac * vec_ab[..., 5, :, :, :]  # 23
    vec_mand[..., 4, :, :, :] = fac * vec_ab[..., 4, :, :, :]  # 31
    vec_mand[..., 5, :, :, :] = fac * vec_ab[..., 3, :, :, :]  # 12

    return vec_mand


class LocalizationDataset(Dataset):
    """Load set of microstructure and strain responses from files"""

    def __init__(self, micro_file, resp_file):
        # store files
        self.mf = micro_file
        self.rf = resp_file

        # handles to datasets directly
        self.micro = None
        self.strain = None
        self.stress = None

        self.metadata_dict = {"phase_info": None, "eps_bar": None}

        self.length = self.getdata(self.mf, dataset_name="micros").shape[0]

        self.constlaw = None

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

            # make sure everything is ok
            assert dataset is not None
            return dataset
        elif ext.lower() in [".npy"]:
            f = np.load(filename)
            return f
        else:
            raise NotImplementedError

    def load_metadata(self, metadata_file):
        # open file
        f = h5py.File(metadata_file, "r", rdcc_nbytes=H5_CACHE_SIZE)
        # load each data for each key directly into memory (should not be that big)
        for key in self.metadata_dict.keys():
            if key in f.keys():
                # load entire metadata into memory (should be small by definition)
                self.metadata_dict[key] = torch.from_numpy(f[key][:])

        # if we got phase info, immediately postprocess into stiffness tensors
        if self.metadata_dict["phase_info"] is not None:
            unique_vals = torch.unique(self.metadata_dict["phase_info"])

            # how many different contrast ratios do we have?
            num_unique = len(unique_vals)

            # build lookup table for a given phase assignment
            self.C_mat_lookup = {}
            # list of sets of stiffness tensors
            self.C_mats = torch.zeros(num_unique, 2, 6, 6)

            for ind, v in enumerate(unique_vals):
                # first dump into tuple so it can act as dict key
                vals = tuple(v.flatten().to_list())

                # now build stiffness mat
                if True:
                    # do 2-phase case for now
                    # separate out values
                    E_0, nu_0, E_1, nu_1 = vals

                    C_mats = torch.zeros(2, 6, 6)

                    C_mats[0] = isotropic_mandel66(YMP_to_Lame(E_0, nu_0))
                    C_mats[1] = isotropic_mandel66(YMP_to_Lame(E_1, nu_1))

                    # store these stiffness matrices at correct index
                    self.C_mats[ind] = C_mats

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

        # pull in avg strain from metadata
        if self.metadata_dict["eps_bar"] is not None:
            eps_bar = self.metadata_dict["eps_bar"][index]

        # if self.metadata_dict['phase_info'] is not None:
        # if self.meta

        # Load data and get label
        micro = torch.from_numpy(self.micro[index]).float()

        # if we have constlaw, convert directly to stiffness
        if self.constlaw is not None:
            # do we have phase info for each micro?
            if self.metadata_dict["phase_info"] is not None:
                pass
                # first construct (retrieve?) phase-wise stiffness, then broadcast

        strain = torch.from_numpy(self.strain[index]).float()
        strain = abaqus_to_mandel(strain, fac=FAC_STRAIN)

        stress = torch.from_numpy(self.stress[index]).float()
        stress = abaqus_to_mandel(stress, fac=FAC_STRESS)

        return micro, strain, stress


#
