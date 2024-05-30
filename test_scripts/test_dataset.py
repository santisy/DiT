import torch
import sys
sys.path.insert(0, ".")
from data_extensions import load_utils
from data.ofalg_dataset import OFLAGDataset

dataset = OFLAGDataset("datasets/shapenet_airplane_l_corrected",
                       octree_root_num=64)
x0, x1, x2, _, _, _, _ = dataset[0]
print(x0.shape)
print(x1.shape)
print(x2.shape)
print("Checking")

x0_out = torch.zeros_like(x0)
x0_out[:, -7] = x0[:, -7]
x0_out[:, -3:] = x0[:, -3:]
x0 = x0_out.clone()

# We do not use any information from l1
x1 = torch.zeros_like(x1)

x0 = dataset.denormalize(x0, 0)
x1 = dataset.denormalize(x1, 1)
x2 = dataset.denormalize(x2, 2)

load_utils.dump_to_bin("test.bin", x0, x1, x2, dataset.octree_root_num)
