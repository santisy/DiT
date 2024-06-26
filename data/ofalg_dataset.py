import json
import glob
import os

from torch.utils.data import Dataset
from data_extensions import load_utils


class OFLAGDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 octree_root_num: int=64,
                 unit_length_list = [361, 139, 139],
                 only_infer=False,
                 ** kwargs):
        super().__init__()

        self._octree_root_num = octree_root_num
        self._unit_length0 = unit_length_list[0]
        self._unit_length1 = unit_length_list[1]
        self._unit_length2 = unit_length_list[2]

        stats_file = os.path.join(data_root, "stats.json")
        assert os.path.isfile(stats_file)
        with open(stats_file, "r") as f:
            self._stats = json.load(f)

        file_paths = glob.glob(os.path.join(data_root, "*.bin"))
        if not only_infer:
            file_paths = [file for file in file_paths if os.path.getsize(file) > 1 * 1024 * 1024]
        self.file_paths = file_paths

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

    @property
    def octree_root_num(self):
        return self._octree_root_num

    def rescale_voxel_len(self, x):
        return x * self._stats["abs_s_0_std"] + self._stats["abs_s_0_mean"]
    
    def rescale_positions(self, x):
        return x * self._stats["abs_p_0_std"] + self._stats["abs_p_0_mean"]

    def denormalize(self, x, l):
        if l == 0:
            j = 0
            x[:, j:j + 7 ** 3] = x[:, j:j + 7 ** 3] * self._stats["grid_0_std"] + self._stats["grid_0_mean"] 
            j += 7 ** 3
            x[:, j:j + 6] = x[:, j:j + 6] * self._stats["ori_0_std"] + self._stats["ori_0_mean"]
            j += 6
            x[:, j:j + 3] = x[:, j:j + 3] * self._stats["rel_half_s_0_std"] + self._stats["rel_half_s_0_mean"]
            j += 3
            x[:, j:j + 1] = x[:, j:j + 1] * self._stats["abs_s_0_std"] + self._stats["abs_s_0_mean"]
            j += 1
            x[:, j:j + 3] = x[:, j:j + 3] * self._stats["rel_p_0_std"] + self._stats["rel_p_0_mean"]
            j += 3
            x[:, j:j + 3] = x[:, j:j + 3] * self._stats["abs_p_0_std"] + self._stats["abs_p_0_mean"]
        else:
            j = 0
            x[:, j:j + 5 ** 3] = x[:, j:j + 5 ** 3] * self._stats[f"grid_{l}_std"] + self._stats[f"grid_{l}_mean"]
            j += 5 ** 3
            x[:, j:j + 6] = x[:, j:j + 6] * self._stats[f"ori_{l}_std"] + self._stats[f"ori_{l}_mean"]
            j += 6
            x[:, j:j + 3] = x[:, j:j + 3] * self._stats[f"rel_half_s_{l}_std"] + self._stats[f"rel_half_s_{l}_mean"]
            j += 3
            x[:, j:j + 3] = x[:, j:j + 3] * self._stats[f"rel_p_{l}_std"] + self._stats[f"rel_p_{l}_mean"]

        return x

    def normalize(self, x, l):
        if l == 0:
            j = 0
            x[:, j:j + 7 ** 3] = (x[:, j:j + 7 ** 3] - self._stats["grid_0_mean"]) / self._stats["grid_0_std"]
            j += 7 ** 3
            x[:, j:j + 6] = (x[:, j:j + 6] - self._stats["ori_0_mean"]) / self._stats["ori_0_std"]
            j += 6
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["rel_half_s_0_mean"]) / self._stats["rel_half_s_0_std"]
            j += 3
            x[:, j:j + 1] = (x[:, j:j + 1] - self._stats["abs_s_0_mean"]) / self._stats["abs_s_0_std"]
            j += 1
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["rel_p_0_mean"]) / self._stats["rel_p_0_std"]
            j += 3
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["abs_p_0_mean"]) / self._stats["abs_p_0_std"]
        else:
            j = 0
            x[:, j:j + 5 ** 3] = (x[:, j:j + 5 ** 3] - self._stats[f"grid_{l}_mean"]) / self._stats[f"grid_{l}_std"]
            j += 5 ** 3
            x[:, j:j + 6] = (x[:, j:j + 6] - self._stats[f"ori_{l}_mean"]) / self._stats[f"ori_{l}_std"]
            j += 6
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_half_s_{l}_mean"]) / self._stats[f"rel_half_s_{l}_std"]
            j += 3
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_p_{l}_mean"]) / self._stats[f"rel_p_{l}_std"]


    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        level0_tensor, level1_tensor, level2_tensor, \
        level0_position, level1_position, level2_position \
            = load_utils.load(file_path,
                              self._unit_length0,
                              self._unit_length1,
                              self._unit_length2)

        assert level0_tensor.size(0) == self._octree_root_num 


        self.normalize(level0_tensor, 0)
        self.normalize(level1_tensor, 1)
        self.normalize(level2_tensor, 2)

        # Dummy label
        label = -1

        return level0_tensor, level1_tensor, level2_tensor, \
               level0_position, level1_position, level2_position, label
