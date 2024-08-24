import json
import io
import glob
import math
import os

from torch.utils.data import Dataset
from data_extensions import load_utils
from utils.parallelzipfile import ParallelZipFile as ZipFile


class OFLAGDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 octree_root_num: int=64,
                 unit_length_list = [361, 139],
                 only_infer=False,
                 validate_num=0,
                 validate_flag=False,
                 **kwargs):
        super().__init__()

        assert data_root.endswith(".zip")

        self._octree_root_num = octree_root_num
        self._unit_length0 = unit_length_list[0]
        self._unit_length1 = unit_length_list[1]
        self._path = data_root
        self._zipfile = None

        all_fnames = self._get_zipfile().namelist()
        all_finfo = self._get_zipfile().infolist()

        json_path = None
        file_paths = []
        label_count = 0
        self.label_dict = {}
        for fname, finfo in zip(all_fnames, all_finfo):
            if os.path.basename(fname) == "stats.json":
                json_path = fname
                continue
            if os.path.basename(fname).endswith(".bin") and finfo.file_size > 1 * 1024 * 1024:
                class_name = os.path.basename(fname).split("_")[0]
                if class_name not in self.label_dict:
                    self.label_dict[class_name] = label_count
                    label_count += 1
                file_paths.append(fname)

        if json_path is None:
            raise RuntimeError(f"No stats.json found in the zip file.")
        with self._open_file(json_path) as f:
            self._stats = json.load(f)

        #if not only_infer:
        #    file_paths = [file for file in file_paths if os.path.getsize(file) > 1 * 1024 * 1024]

        # Split the dataset to validate one if required
        if validate_flag:
            assert validate_num > 0
            self.file_paths = file_paths[-validate_num:]
        else:
            if validate_num > 0:
                self.file_paths = file_paths[:-validate_num]
            else:
                self.file_paths = file_paths

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        return io.BytesIO(self._get_zipfile().read(fname))

    def _open_bytes(self, fname):
        return self._get_zipfile().read(fname)

    def __len__(self):
        return len(self.file_paths)

    @property
    def class_num(self):
        return len(self.label_dict)

    def get_ref_objects(self):
        pass

    def get_sample_num(self):
        return math.floor(len(self) // 4 * 0.05)

    def get_level_vec_len(self, level_num):
        if level_num == 0:
            return self._unit_length0
        elif level_num == 1:
            return self._unit_length1 - 4
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
    
    def get_condition_dim(self, level_num, sibling_num=2, no_a_flag=False):
        if level_num == 0:
            return []
        elif level_num == 1:
            return [4,]
        elif level_num == 2:
            if not no_a_flag:
                return [4, 10 * sibling_num]
            else:
                return [10 * sibling_num,]
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
            # Use Quaternions instead of others
            x[:, j:j + 4] = x[:, j:j + 4] * 2.0 - 1.0
            j += 4
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
            x[:, j:j + 4] = x[:, j:j + 4] * 2.0 - 1.0
            j += 4
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats[f"rel_half_s_{l}_max"] - self._stats[f"rel_half_s_{l}_min"]) + self._stats[f"rel_half_s_{l}_min"]
            j += 3
            x[:, j:j + 3] = x[:, j:j + 3] * (self._stats[f"rel_p_{l}_max"] - self._stats[f"rel_p_{l}_min"]) + self._stats[f"rel_p_{l}_min"]

        return x

    def normalize(self, x, l):
        if l == 0:
            j = 0
            x[:, j:j + 7 ** 3] = (x[:, j:j + 7 ** 3] - self._stats["grid_0_min"]) / (self._stats["grid_0_max"] - self._stats["grid_0_min"])
            j += 7 ** 3
            x[:, j:j + 4] = (x[:, j:j + 4] + 1.0) / 2.0
            j += 4
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
            x[:, j:j + 4] = (x[:, j:j + 4] + 1.0) / 2.0
            j += 4
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_half_s_{l}_min"]) / (self._stats[f"rel_half_s_{l}_max"] - self._stats[f"rel_half_s_{l}_min"])
            j += 3
            x[:, j:j + 3] = (x[:, j:j + 3] - self._stats[f"rel_p_{l}_min"]) / (self._stats[f"rel_p_{l}_max"] - self._stats[f"rel_p_{l}_min"])


    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        class_name = os.path.basename(file_path).split("_")[0]

        level0_tensor, level1_tensor, \
        level0_position, level1_position \
            = load_utils.load(self._open_bytes(file_path),
                              self._unit_length0,
                              self._unit_length1)

        assert level0_tensor.size(0) == self._octree_root_num 


        self.normalize(level0_tensor, 0)
        self.normalize(level1_tensor, 1)

        # Dummy label
        label = self.label_dict[class_name]

        return level0_tensor, level1_tensor,  \
               level0_position, level1_position, label
