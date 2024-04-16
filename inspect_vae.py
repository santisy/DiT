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
import numpy as np
from tqdm import tqdm

from vae_model import VAE, OnlineVariance
import argparse
from data.ofalg_dataset import OFLAGDataset

def main(args):
    # Prepare name and directory
    out_dir = args.export_out
    os.makedirs(out_dir, exist_ok=True)
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
    for l in range(3):
        in_ch = dataset.get_level_vec_len(l)
        latent_dim = in_ch // config.vae.latent_ratio
        hidden_size = int(in_ch * 8)
        vae_model = VAE(config.vae.layer_num,
                        in_ch,
                        hidden_size,
                        latent_dim)
        vae_model.load_state_dict(vae_ckpt["model"][l])
        vae_model = vae_model.to(device)
        vae_model_list.append(vae_model)
        online_variance_list.append(OnlineVariance(latent_dim))
    vae_model_list.eval()


    # Go through the whole dataset
    for i in tqdm(range(len(dataset))):
        x0, x1, x2, _, _, _, _ = dataset[i]
        x0 = x0.unsqueeze(dim=0).to(device)
        x1 = x1.unsqueeze(dim=0).to(device)
        x2 = x2.unsqueeze(dim=0).to(device)
        with torch.no_grad():
            x0_rec, _, _ = vae_model_list[0](x0)
            x1_rec, _, _ = vae_model_list[1](x1)
            x2_rec, _, _ = vae_model_list[2](x2)
            loss = F.mse_loss(x0_rec, x0, reduction="sum")/ x0.size(1) + \
                   F.mse_loss(x1_rec, x1, reduction="sum")/ x1.size(1) + \
                   F.mse_loss(x2_rec, x2, reduction="sum")/ x2.size(1)

            loss_l1 = F.l1_loss(x0_rec, x0, reduction="sum")/ x0.size(1) + \
                   F.l1_loss(x1_rec, x1, reduction="sum")/ x1.size(1) + \
                   F.l1_loss(x2_rec, x2, reduction="sum")/ x2.size(1)

            print(loss, loss_l1)
            import pdb; pdb.set_trace()

            #latent_0 = vae_model_list[0].encode_and_reparam(x0)
            #latent_1 = vae_model_list[1].encode_and_reparam(x1)
            #latent_2 = vae_model_list[2].encode_and_reparam(x2)
            #online_variance_list[0].update(latent_0[0].detach().cpu())
            #online_variance_list[1].update(latent_1[0].detach().cpu())
            #online_variance_list[2].update(latent_2[0].detach().cpu())

    # Dump the statistics
    np.savez(os.path.join(out_dir, f"{ckpt_name}-{dataset_name}-stds"),
             std0=online_variance_list[0].std,
             std1=online_variance_list[1].std,
             std2=online_variance_list[2].std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--export-out", type=str, required=True)
    args = parser.parse_args()
    main(args)
