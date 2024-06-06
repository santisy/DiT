import torch
import sys
sys.path.insert(0, ".")
from data_extensions import load_utils
from data.ofalg_dataset import OFLAGDataset

dataset = OFLAGDataset("datasets/shapenet_airplane_l1only_abs", octree_root_num=256)
x0, x1, _, _, _ = dataset[200]
print(x0.shape)
print(x1.shape)
print("Checking")

x0_out = torch.zeros_like(x0)
x0_out[:, -7] = x0[:, -7]
x0_out[:, -3:] = x0[:, -3:]
x0 = x0_out.clone()
x0 = dataset.denormalize(x0, 0)
x1 = dataset.denormalize(x1, 1)

load_utils.dump_to_bin("test.bin", x0, x1, dataset.octree_root_num)
