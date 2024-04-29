import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
import numpy as np
import h5py
import math

# increase cache to 1GB
H5_CACHE_SIZE = 4 * 1024 * 1024 * 1024

SQRT2 = math.sqrt(2.0)


def abaqus_to_mandel(vec_ab):
    # return vec_ab
    # print(vec_ab.shape, vec_ab[:, 0].shape)
    vec_mand = vec_ab.new_zeros(vec_ab.shape)
    vec_mand[..., 0, :, :, :] = vec_ab[..., 0, :, :, :]  # 11
    vec_mand[..., 1, :, :, :] = vec_ab[..., 1, :, :, :]  # 22
    vec_mand[..., 2, :, :, :] = vec_ab[..., 2, :, :, :]  # 33

    vec_mand[..., 3, :, :, :] = SQRT2 * vec_ab[..., 5, :, :, :]  # 23
    vec_mand[..., 4, :, :, :] = SQRT2 * vec_ab[..., 4, :, :, :]  # 31
    vec_mand[..., 5, :, :, :] = SQRT2 * vec_ab[..., 3, :, :, :]  # 12

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

        self.length = self.getdata(self.mf, dataset_name="micros").shape[0]
        # self.length = 128

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

        # Load data and get label
        X = torch.from_numpy(self.micro[index]).float()

        strain = abaqus_to_mandel(torch.from_numpy(self.strain[index]).float())
        stress = abaqus_to_mandel(torch.from_numpy(self.stress[index]).float())

        return X, strain, stress
