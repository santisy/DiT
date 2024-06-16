import torch
import sys
from ruamel.yaml import YAML
from easydict import EasyDict as edict

sys.path.insert(0, ".")

from data_extensions import load_utils
from data.ofalg_dataset import OFLAGDataset
from diffusion import create_diffusion
from train import noise_conditioning

with open("configs/OFALG_config_v9_ag_nl_small.yaml", "r") as f:
    yaml = YAML()
    config = edict(yaml.load(f))
sampler = create_diffusion(timestep_respacing="", **config.diffusion)

dataset = OFLAGDataset("datasets/shapenet_airplane_l1only_abs", octree_root_num=256)
x0, x1, _, _, _ = dataset[200]
#print(x0.shape)
#print(x1.shape)
#print("Checking")

#x0_out = torch.zeros_like(x0)
#x0_out[:, -7] = x0[:, -7]
#x0_out[:, -3:] = x0[:, -3:]
#x0 = x0_out.clone()
#x0 = dataset.denormalize(x0, 0)
#x1 = dataset.denormalize(x1, 1)
#load_utils.dump_to_bin("test.bin", x0, x1, dataset.octree_root_num)

x1_raw = x1[:, 125:].unsqueeze(dim=0)
x1_list = [x1_raw.clone(),]
a = [torch.tensor((20,)).long(),]
x1_list = noise_conditioning(x1_list, a, sampler)
import pdb; pdb.set_trace()
print("Checking")
