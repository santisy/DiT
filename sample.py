# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import math
import torch.nn as nn
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from ruamel.yaml import YAML
from easydict import EasyDict as edict
from models import DiT
import numpy as np

from vae_model import VAE, VAELinear
import argparse
from data.ofalg_dataset import OFLAGDataset
from data_extensions import load_utils


def main(args):
    # Make directories
    os.makedirs(args.export_dir, exist_ok=True)
    out_dir = args.export_dir

    # Load config
    with open(args.config_file, "r") as f:
        yaml = YAML()
        config = edict(yaml.load(f))

    # Create dataset. For denormalizing
    dataset = OFLAGDataset(args.data_root, only_infer=True, **config.data)

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # VAE model and stats loading
    vae_model_list = nn.ModuleList()
    vae_ckpt = torch.load(args.vae_ckpt, map_location=lambda storage, loc: storage)
    vae_std_loaded = np.load(args.vae_std)
    vae_std_list = [
                    torch.from_numpy(vae_std_loaded["std2"]).unsqueeze(dim=0).unsqueeze(dim=0).clone().to(device),
                    ]

    linear_flag = config.vae.linear
    for l in range(2, 3):
        in_ch = dataset.get_level_vec_len(l)
        m = int(math.floor(math.pow(in_ch, 1 / 3.0)))
        if not linear_flag:
            model = VAE(config.vae.layer_n,
                        config.vae.in_ch,
                        config.vae.latent_ch,
                        m)
        else:
            in_ch = int(m ** 3)
            latent_dim = in_ch // config.vae.latent_ratio
            model = VAELinear(config.vae.layer_n,
                              in_ch,
                              in_ch * 16,
                              latent_dim)
        vae_model.load_state_dict(vae_ckpt["model"][l - 2])
        vae_model = vae_model.to(device)
        vae_model_list.append(vae_model)
    vae_model_list.eval()

    # Model
    model_list = []
    in_ch_list = []
    m = None
    m_ = None
    for l in range(3):
        num_heads = config.model.num_heads
        depth = config.model.depth
        if l == 2:
            in_ch = dataset.get_level_vec_len(2)
            m_ = math.ceil(m / 2)
            in_ch = int(m_ ** 3 * config.vae.latent_ch)
            in_ch_list.append(in_ch)
            hidden_size = 1024
        elif l == 1: # Leaf 
            # Length 14: orientation 8 + scales 3 + relative positions 3
            in_ch = int(dataset.get_level_vec_len(2) - m ** 3)
            in_ch_list.append(in_ch)
            hidden_size = 512
        elif l == 0: # Root positions and scales
            in_ch = 4
            in_ch_list.append(in_ch)
            hidden_size = 1024

        # Create DiT model
        model = DiT(
            # Data related
            in_channels=in_ch, # Combine to each children
            num_classes=config.data.num_classes,
            condition_node_num=dataset.get_condition_num(l),
            condition_node_dim=dataset.get_condition_dim(l),
            # Network itself related
            hidden_size=hidden_size, # 4 times rule
            mlp_ratio=config.model.mlp_ratio,
            depth=depth,
            num_heads=num_heads,
            learn_sigma=config.diffusion.get("learn_sigma", True),
            # Other flags
            add_inject=config.model.add_inject,
            aligned_gen=True if l != 0 else False,
            pos_embedding_version=config.model.get("pos_emedding_version", "v1"),
            l=l
        )
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = args.ckpt[l]
        print(f"\033[92mLoading model level {l}: {ckpt_path}.\033[00m")
        model_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_ckpt["model"])
        model.eval()  # important!
        model_list.append(model)

    diffusion = create_diffusion(timestep_respacing="",
                                 **config.diffusion)

    batch_size = args.sample_batch_size
    sample_num = args.sample_num
    for i in range(sample_num // batch_size):
        xc = []
        positions = []
        scales = []
        decoded = []
        for l in range(3):
            length = dataset.octree_root_num * 8 if l == 0 else dataset.octree_root_num * 8 ** 2
            ch = in_ch_list[l]
            z = torch.randn(batch_size,
                            length, 
                            ch).to(device)
            a = [torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device) for _ in range(l)]
            model_kwargs = dict(a=a, y=None, x0=xc, positions=positions)

            # Sample
            samples = diffusion.p_sample_loop(model_list[l].forward,
                                              z.shape,
                                              z,
                                              model_kwargs=model_kwargs,
                                              clip_denoised=False,
                                              progress=False,
                                              device=device)

            # Append the generated latents for the following generation
            if l in [0, 1]:
                samples = samples.clip_(0, 1)
            xc.append(samples.clone())

            if l == 2:
                # Rescale and decode
                samples = samples * vae_std_list[0]
                samples = samples.reshape(batch_size * length, config.vae.latent_ch, m_, m_, m_)
                samples = vae_model_list[0].decode(samples)
                samples = samples.reshape(batch_size, length, -1)
                decoded.append(torch.cat([samples, xc[-1]], dim=-1).clone())
            elif l == 0:
                sample_ = torch.zeros(batch_size, length, dataset.get_level_vec_len(0)).to(device)
                sample_[:, :, -7] = samples[:, :, 0]
                sample_[:, :, -3:] = samples[:, :, -3:]
                decoded.append(sample_.clone())

            # Get the positions and scales from generated
            scales.append(None)
            positions.append(None)

        # Denormalize and dump
        for j in range(batch_size):
            x0 = dataset.denormalize(decoded[0][j], 0).detach().cpu()
            x1 = dataset.denormalize(torch.zeros((dataset.octree_root_num * 8,
                                                  dataset.get_level_vec_len(1)),
                                                  dtype=torch.float32, device=device)).detach().cpu()
            x2 = dataset.denormalize(decoded[1][j], 2).detach().cpu()
            load_utils.dump_to_bin(os.path.join(out_dir, f"out_{j + i * batch_size:04d}.bin"),
                                   x0, x1, x2, dataset.octree_root_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", nargs='+',
                        help="A list of strings that provides three level generation models.")
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--sample-num", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--vae-std", type=str, required=True)
    args = parser.parse_args()
    main(args)
