import sys
sys.path.insert(0, ".")
from data.ofalg_dataset import OFLAGDataset

dataset = OFLAGDataset("datasets/shapenet_airplane")
x0, x1, x2, _, _, _, _ = dataset[0]
print(x0.shape)
print(x1.shape)
print(x2.shape)
import pdb; pdb.set_trace()
print("Checking")
