import sys
from data_extensions import load_utils
sys.path.insert(0, ".")
from data.ofalg_dataset import OFLAGDataset

dataset = OFLAGDataset("datasets/shapenet_airplane")
x0, x1, _, _, _ = dataset[0]
print(x0.shape)
print(x1.shape)
print("Checking")

x0 = dataset.denormalize(x0, 0)
x1 = dataset.denormalize(x1, 1)

load_utils.dump_to_bin("test.bin", x0, x1, dataset.octree_root_num)
