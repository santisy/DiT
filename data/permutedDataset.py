import torch
from torch.utils.data import Dataset

import numpy as np


class MNISTPermutedDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = np.fromfile("datasets/data.bin", dtype=np.float32)
        self.data = data.reshape(60000, 240)

    def __len__(self):
        return 60000

    def __getitem__(self, idx):

        level0 = self.data[idx][:48].copy().reshape(16, 3)
        level1 = self.data[idx][48:].copy().reshape(64, 3)

        level0_new = np.zeros((16, 3), dtype=np.float32)
        level1_new = np.zeros((64, 3), dtype=np.float32)

        node0_indices = np.where(level0.sum(axis=1) != 0)[0]
        node0_new_indices = np.arange(len(node0_indices))
        permuted_indices0 = np.arange(len(node0_indices))
        np.random.shuffle(permuted_indices0)
        level0_new[:len(node0_indices), :] = level0[node0_indices, :]
        permuted_level0 = level0_new[permuted_indices0, :].copy()
        level0_new[:len(node0_indices), :] = permuted_level0

        # Put non-zero items at first
        node1_indices = np.where(level1.sum(axis=1) != 0)[0]
        level1_new[:len(node1_indices), :] = level1[node1_indices, :]
        # Permute level1
        permuted_indices1 = np.arange(len(node1_indices))
        np.random.shuffle(permuted_indices1)
        permuted_level1 = level1_new[permuted_indices1, :].copy()
        level1_new[:len(node1_indices)] = permuted_level1

        # Change the parent indices because of the permutation
        for i in range(64):
            if level1_new[i][2] != 0:
                ori_pIdx = int(level1_new[i][0] * 16.0)
                before_permute_idx = np.where(node0_indices == ori_pIdx)[0]
                after_permute_idx = np.where(permuted_indices0 == before_permute_idx)[0]
                level1_new[i][0] = float(after_permute_idx) / 16.0

        #NOTE: From [0, 1] to [-1, 1]
        level0_new = level0_new * 2.0 - 1.0
        level1_new = level1_new * 2.0 - 1.0
        label = -1 # Class label

        return level0_new, level1_new, label
