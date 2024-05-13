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
                 validate_num=0,
                 validate_flag=False,
                 **kwargs):
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

        # Split the dataset to validate one if required
        if validate_flag:
            assert validate_num > 0
            self.file_paths = file_paths[-validate_num:]
        else:
            if validate_num > 0:
                self.file_paths = file_paths[:-validate_num]
            else:
                self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def get_level_vec_len(self, level_num):
        if level_num == 0:
            return self._unit_length0
        elif level_num == 1:
            return self._unit_length1
        elif level_num == 2:
            return self._unit_length2
        else:
            raise ValueError(f"Invalid level number {level_num}.")

    def get_condition_num(self, level_num):
        if level_num == 0:
            return []
        elif level_num == 1:
            return [self._octree_root_num]
        elif level_num == 2:
            return [self._octree_root_num, self._octree_root_num * 8]
        else:
            raise ValueError(f"Invalid level number {level_num}.")
    
    def get_condition_dim(self, level_num):
        if level_num == 0:
            return []
        elif level_num == 1:
            return [4,]
        elif level_num == 2:
            return [4, 28]
        else:
            raise ValueError(f"Invalid level number {level_num}.")

    @property
    def octree_root_num(self):
        return self._octree_root_num

    def rescale_voxel_len(self, x):
        return x * (self._stats["abs_s_0_max"] - self._stats["abs_s_0_min"]) + self._stats["abs_s_0_min"]
    
    def rescale_positions(self, x):
        return x * (self._stats["abs_p_0_max"] - self._stats["abs_p_0_min"]) + self._stats["abs_p_0_min"]

    def denormalize(self, x, l):
        if l == 0:
            j = 0
            x[:, j:j + 7 ** 3] = x[:, j:j + 7 ** 3] * (self._stats["grid_0_max"] - self._stats["grid_0_min"])  + self._stats["grid_0_min"]
            j += 7 ** 3
            # Angular encoding orientations
            x[:, j:j + 8] = x[:, j:j + 8] * 2.0 - 1.0
            j += 8 
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats["rel_half_s_0_max"] - self._stats["rel_half_s_0_min"]) + self._stats["rel_half_s_0_min"]
            j += 3
            x[:, j:j + 1] = x[:, j:j + 1] * (self._stats["abs_s_0_max"] - self._stats["abs_s_0_min"]) + self._stats["abs_s_0_min"]
            j += 1
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats["rel_p_0_max"] - self._stats["rel_p_0_min"]) + self._stats["rel_p_0_min"]
            j += 3
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats["abs_p_0_max"] - self._stats["abs_p_0_min"]) + self._stats["abs_p_0_min"]
        else:
            j = 0
            x[:, j:j + 5 ** 3] = x[:, j:j + 5 ** 3] * (self._stats[f"grid_{l}_max"] - self._stats[f"grid_{l}_min"]) + self._stats[f"grid_{l}_min"]
            j += 5 ** 3
            x[:, j:j + 8] = x[:, j:j + 8] * 2.0 - 1.0
            j += 8
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats[f"rel_half_s_{l}_max"] - self._stats[f"rel_half_s_{l}_min"]) + self._stats[f"rel_half_s_{l}_min"]
            j += 3
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats[f"rel_p_{l}_max"] - self._stats[f"rel_p_{l}_min"]) + self._stats[f"rel_p_{l}_min"]

        return x

    def normalize(self, x, l):
        if l == 0:
            j = 0
            x[:, j:j + 7 ** 3] = (x[:, j:j + 7 ** 3] - self._stats["grid_0_min"]) / (self._stats["grid_0_max"] - self._stats["grid_0_min"])
            j += 7 ** 3
            x[:, j:j + 8] = (x[:, j:j + 8] + 1.0) / 2.0
            j += 8
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["rel_half_s_0_min"]) / (self._stats["rel_half_s_0_max"] - self._stats["rel_half_s_0_min"])
            j += 3
            x[:, j:j + 1] = (x[:, j:j + 1] - self._stats["abs_s_0_min"]) / (self._stats["abs_s_0_max"] - self._stats["abs_s_0_min"])
            j += 1
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["rel_p_0_min"]) / (self._stats["rel_p_0_max"] - self._stats["rel_p_0_min"])
            j += 3
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats["abs_p_0_min"]) / (self._stats["abs_p_0_max"] - self._stats["abs_p_0_min"])
        else:
            j = 0
            x[:, j:j + 5 ** 3] = (x[:, j:j + 5 ** 3] - self._stats[f"grid_{l}_min"]) / (self._stats[f"grid_{l}_max"] - self._stats[f"grid_{l}_min"])
            j += 5 ** 3
            x[:, j:j + 8] = (x[:, j:j + 8] + 1.0) / 2.0
            j += 8
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_half_s_{l}_min"]) / (self._stats[f"rel_half_s_{l}_max"] - self._stats[f"rel_half_s_{l}_min"])
            j += 3
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_p_{l}_min"]) / (self._stats[f"rel_p_{l}_max"] - self._stats[f"rel_p_{l}_min"])


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
