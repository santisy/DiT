import sys

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML
from easydict import EasyDict as edict
import numpy as np

sys.path.insert(0, ".")

from data_extensions import load_utils
from data.ofalg_dataset import OFLAGDataset
from diffusion import create_diffusion
from train import noise_conditioning
from plot_dir.plot_tool import plot_root_aabb

#with open("configs/OFALG_config_v9_ag_nl_small.yaml", "r") as f:
#    yaml = YAML()
#    config = edict(yaml.load(f))
#sampler = create_diffusion(timestep_respacing="", **config.diffusion)

dataset = OFLAGDataset("/media/dya62/Data2/datasets/shapenetManifold/shapenet_airplane_discreteL1.zip", octree_root_num=256)

#x0, x1, _, _, label = dataset[200]
#print(x0.shape)
#print(x1.shape)
#print("Checking")

loader = DataLoader(
    dataset,
    batch_size=6,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
)

for x0, x1, _, _, y in loader:
    print(x0.shape)
    print(x1.shape)
    print(y)
    print(y.shape)
    print(type(y))
    exit()



## Visualize the Grid to see the data loading correction or not
#x0 = dataset.denormalize(x0, 0)
#data = x0.numpy()
#data = np.concatenate([data[:, -7][:, None], data[:, -3:]], axis=1)
#data = data.tolist()
#plot_root_aabb(data)

## Test dumping to binary
#x0_out = torch.zeros_like(x0)
#x0_out[:, -7] = x0[:, -7]
#x0_out[:, -3:] = x0[:, -3:]
#x0 = x0_out.clone()
#x1 = dataset.denormalize(x1, 1)
#load_utils.dump_to_bin("test.bin", x0, x1, dataset.octree_root_num)

## Test noise conditioning
#x1_raw = x1[:, 125:].unsqueeze(dim=0)
#x1_list = [x1_raw.clone(),]
#a = [torch.tensor((20,)).long(),]
#x1_list = noise_conditioning(x1_list, a, sampler)
#import pdb; pdb.set_trace()
#print("Checking")
