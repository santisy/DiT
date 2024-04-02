import glob
import os

from torch.utils.data import Dataset

from data_extensions import load_utils


class OFLAGDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 octree_root_num: int=64,
                 unit_length_list = [361, 139, 139],
                 ** kwargs):
        super().__init__()

        self._octree_root_num = octree_root_num
        self._unit_length0 = unit_length_list[0]
        self._unit_length1 = unit_length_list[1]
        self._unit_length2 = unit_length_list[2]

        self._max_voxel_len = load_utils.max_voxel_length(data_root, self._unit_length0)
        self.file_paths = glob.glob(os.path.join(data_root, "*.bin"))

    def __len__(self):
        return len(self.file_paths)

    def get_level_vec_len(self, level_num):
        if level_num == 0:
            return self._unit_length0 - 2
        elif level_num == 1:
            return self._unit_length1 - 2
        elif level_num == 2:
            return self._unit_length2 - 2
        else:
            raise ValueError(f"Invalid level number {level_num}.")

    def get_condition_num(self, level_num):
        if level_num == 0:
            return []
        elif level_num == 1:
            return [self._octree_root_num,]
        elif level_num == 2:
            return [self._octree_root_num, self._octree_root_num * 8]
        else:
            raise ValueError(f"Invalid level number {level_num}.")
    
    def get_condition_dim(self, level_num):
        if level_num == 0:
            return []
        elif level_num == 1:
            return [self._unit_length0 - 2,]
        elif level_num == 2:
            return [self._unit_length0 - 2, self._unit_length1 - 2]
        else:
            raise ValueError(f"Invalid level number {level_num}.")

    def denormalize(self, x, level_num):
        # Make sure normalize everything about to [-1.0, 1.0]
        # Normalize grids
        vec_len = self.get_level_vec_len(level_num)
        if level_num == 0:
            x[:, :7 ** 3] *= 0.1
            x[:, vec_len - 7] = (x[:, vec_len - 7] + 1.0) / 2.0
            x[:, vec_len - 7] *= self._max_voxel_len
            x[:, vec_len - 6:vec_len - 3] = (x[:, vec_len - 6:vec_len - 3] + 1.0) / 2.0
        else:
            x[:, :5 ** 3] *= 0.1
            x[:, vec_len - 3:] = (x[:, vec_len - 3:] + 1.0) / 2.0

        return x


    def __getitem__(self, idx):
        #TODO: Get positional embeddings
        file_path = self.file_paths[idx]
        level0_tensor, level1_tensor, level2_tensor, \
        level0_position, level1_position, level2_position \
            = load_utils.load(file_path,
                              self._unit_length0,
                              self._unit_length1,
                              self._unit_length2)

        assert level0_tensor.size(0) == self._octree_root_num


        level0_vec_len = self.get_level_vec_len(0)
        level1_vec_len = self.get_level_vec_len(1)
        level2_vec_len = self.get_level_vec_len(2)

        # Make sure normalize everything about to [-1.0, 1.0]
        # Normalize grids
        level0_tensor[:, :7 ** 3] /= 0.1
        level1_tensor[:, :5 ** 3] /= 0.1
        level2_tensor[:, :5 ** 3] /= 0.1
        # Normalize absolute voxel lengths at level 0 (tree root level)
        level0_tensor[:, level0_vec_len - 7] /= self._max_voxel_len
        level0_tensor[:, level0_vec_len - 7] = level0_tensor[:, level0_vec_len - 7] * 2.0 - 1.0
        # Normalize relative scales
        level0_tensor[:, level0_vec_len - 6:level0_vec_len - 3] = level0_tensor[:, level0_vec_len - 6:level0_vec_len - 3] * 2.0 - 1.0
        level1_tensor[:, level1_vec_len - 3:] = level1_tensor[:, level1_vec_len - 3:] * 2.0 - 1.0
        level2_tensor[:, level2_vec_len - 3:] = level2_tensor[:, level2_vec_len - 3:] * 2.0 - 1.0

        # Dummy label
        label = -1

        return level0_tensor, level1_tensor, level2_tensor, \
               level0_position, level1_position, level2_position, label
