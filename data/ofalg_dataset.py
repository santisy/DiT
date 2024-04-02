import glob
import os

from torch.utils.data import Dataset

from data_extensions import load_utils


class OFLAGDataset(Dataset):
    def __init__(self, data_root: str,
                 unit_length0: int=361,
                 unit_length1: int=139,
                 unit_length2: int=139):
        super().__init__()
        self._unit_length0 = unit_length0
        self._unit_length1 = unit_length1
        self._unit_length2 = unit_length2

        self._max_voxel_len = load_utils.max_voxel_length(data_root, self._unit_length0)
        self.file_paths = glob.glob(os.path.join(data_root, "*.bin"))

    def __len__(self):
        return len(self.file_paths)

    @property
    def level0_vec_len(self):
        return self._unit_length0 - 2

    @property
    def level1_vec_len(self):
        return self._unit_length1 - 2

    @property
    def level2_vec_len(self):
        return self._unit_length2 - 2

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        level0_tensor, level1_tensor, level2_tensor = load_utils.load(file_path,
                                                                      self._unit_length0,
                                                                      self._unit_length1,
                                                                      self._unit_length2)

        # Make sure normalize everything about to [-1.0, 1.0]
        # Normalize grids
        level0_tensor[:, :7 ** 3] /= 0.1
        level1_tensor[:, :5 ** 3] /= 0.1
        level2_tensor[:, :5 ** 3] /= 0.1
        # Normalize absolute voxel lengths at level 0 (tree root level)
        level0_tensor[:, self.level0_vec_len - 7] /= self._max_voxel_len

        # Dummy label
        label = -1

        return level0_tensor, level1_tensor, level2_tensor, label
