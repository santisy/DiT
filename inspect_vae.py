# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inspect and collect statistics of vae encoded
"""
import os
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from ruamel.yaml import YAML
from easydict import EasyDict as edict
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import math

from vae_model import VAE, VAELinear, OnlineVariance
import argparse
from data.ofalg_dataset import OFLAGDataset
from data_extensions import load_utils

def main(args):
    # Prepare name and directory
    out_dir = args.export_out
    os.makedirs(out_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.dirname(args.ckpt))
    ckpt_name = os.path.basename(args.ckpt).split(".")[0]
    dataset_name = os.path.basename(args.data_root)

    # Load config
    with open(args.config_file, "r") as f:
        yaml = YAML()
        config = edict(yaml.load(f))

    # Create dataset. For denormalizing
    dataset = OFLAGDataset(args.data_root, only_infer=False, **config.data)

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare VAE model
    vae_model_list = nn.ModuleList()
    online_variance_list = []
    vae_ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    linear_flag = config.vae.linear
    for l in range(2, 3):
        in_ch = dataset.get_level_vec_len(l)
        m = int(math.floor(math.pow(in_ch, 1 / 3.0)))
        if not linear_flag:
            latent_dim = int(math.ceil(m / 2) ** 3 * config.vae.latent_ch)
            vae_model = VAE(config.vae.layer_n,
                        config.vae.in_ch,
                        config.vae.latent_ch,
                        m,
                        quant_code_n=config.vae.get("quant_code_n", 2048),
                        quant_version=config.vae.get("quant_version", "v0"))
        else:
            in_ch = int(m ** 3)
            latent_dim = in_ch // config.vae.latent_ratio
            vae_model = VAELinear(config.vae.layer_n,
                              in_ch,
                              in_ch * 16,
                              latent_dim)
        vae_model.load_state_dict(vae_ckpt["model"][l - 2])
        vae_model = vae_model.to(device)
        vae_model_list.append(vae_model)
        online_variance_list.append(OnlineVariance(latent_dim))
    vae_model_list.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # Go through the whole dataset
    npy_out = os.path.join(out_dir, f"{exp_name}-{ckpt_name}-{dataset_name}-stds")
    if os.path.isfile(npy_out + ".npz"):
        std = torch.from_numpy(np.load(npy_out + ".npz")["std2"]).unsqueeze(dim=0).unsqueeze(dim=0).clone().to(device)
    else:
        std = 1.0
    count = 0
    for x0, x1, x2, _, _, _, _ in tqdm(loader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        with torch.no_grad():
            if args.inspect_recon:
                #x0_rec, _, _ = vae_model_list[0](x0)
                #x1_rec, _, _ = vae_model_list[0](x1)
                x2_other = x2[:, :, m**3:]
                x2 = x2[:, :, :m ** 3]
                with autocast():
                    indices = vae_model_list[0].get_normalized_indices(x2)
                indices = indices.float()
                x2_rec = vae_model_list[0].decode_code(indices)
                ##loss0 = (x0 - x0_rec).abs().mean()
                #loss1 = (x1 - x1_rec).abs() / x1.size(1)
                loss2 = (x2 - x2_rec).abs() / (x2.size(1) * x2.size(0))
                loss_train = loss2.sum()
                print(loss_train)
                import pdb; pdb.set_trace()
                #print("Checking")
                ##x0 = dataset.denormalize(x0_rec[0], 0).detach().cpu()
                #x1 = dataset.denormalize(x1_rec[0], 1).detach().cpu()
                #x2 = dataset.denormalize(x2_rec[0], 2).detach().cpu()
                x0_out = torch.zeros_like(x0)
                x0_out[:, :, -7] = x0[:, :, -7]
                x0_out[:, :, -3:] = x0[:, :, -3:]
                x0 = x0_out.clone()
                #x0 = dataset.denormalize(x0_out.clone(), 0).detach().cpu()
                x2 = torch.cat([x2_rec, x2_other], dim=-1).clone()
                x1 = torch.zeros_like(x1)
                x0 = dataset.denormalize(x0[0], 0).detach().cpu()
                x1 = dataset.denormalize(x1[0], 1).detach().cpu()
                x2 = dataset.denormalize(x2[0], 2).detach().cpu()
                load_utils.dump_to_bin(os.path.join("vae_recon_dir", f"out_{count:04d}.bin"),
                                       x0, x1, x2, dataset.octree_root_num)
                count += 1
            else:
                x2 = x2[:, :, :m ** 3]
                with torch.no_grad():
                    latent_2 = vae_model_list[0].encode_and_reparam(x2)
                B, L, _ = latent_2.shape
                latent_2 = latent_2.reshape(B * L, -1)
                online_variance_list[0].update(latent_2.detach().cpu())

    if not args.inspect_recon:
        # Dump the statistics
        np.savez(npy_out, std2=online_variance_list[0].std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--export-out", type=str, required=True)
    parser.add_argument("--inspect-recon", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args)