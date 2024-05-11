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
from edm import EDMPrecond, EDMLoss


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

    linear_flag = config.vae.linear
    m = None
    for l in range(2, 3):
        in_ch = dataset.get_level_vec_len(l)
        m = int(math.floor(math.pow(in_ch, 1 / 3.0)))
        if not linear_flag:
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
    vae_model_list.eval()

    # Arch variables
    num_heads = config.model.num_heads
    depth = config.model.depth
    hidden_size = config.model.hidden_sizes[2]
    in_ch = int(math.ceil(m / 2) ** 3) + 14 + 4
    edm_flag = config.model.get("use_EDM", False)

    # Create DiT model
    model = DiT(
        # Data related
        in_channels=in_ch, # Combine to each children
        num_classes=config.data.num_classes,
        condition_node_num=[],
        condition_node_dim=[],
        # Network itself related
        hidden_size=hidden_size, # 4 times rule
        mlp_ratio=config.model.mlp_ratio,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=config.diffusion.get("learn_sigma", True),
        # Other flags
        cross_layers=config.model.cross_layers,
        add_inject=config.model.add_inject,
        aligned_gen=True,
        pos_embedding_version=config.model.get("pos_emedding_version", "v1"),
        level_num=2
    ).to(device)
    if edm_flag:
        model = EDMPrecond(model, n_latents=dataset.octree_root_num * 8 ** 2, channels=in_ch)
        edm_loss = EDMLoss()
    else:
        diffusion = create_diffusion(timestep_respacing="",
                                    **config.diffusion)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt[0]
    print(f"\033[92mLoading model level {l}: {ckpt_path}.\033[00m")
    model_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_ckpt["model"])
    model.to(device)
    model.eval()  # important!

    batch_size = args.sample_batch_size
    sample_num = args.sample_num
    for i in range(sample_num // batch_size):
        xc = []
        positions = []
        scales = []
        decoded = []
        a = []

        # Random generator
        seed = i
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Shape parameter
        length = dataset.octree_root_num * 8 ** 2
        ch = in_ch

        # Random input
        z = torch.randn(batch_size,
                        length, 
                        ch,
                        generator=generator,
                        device=device)
        model_kwargs = dict(a=a, y=None, x0=xc, positions=positions)

        # Sample
        if not edm_flag:
            samples = diffusion.p_sample_loop(model.forward,
                                            z.shape,
                                            z,
                                            model_kwargs=model_kwargs,
                                            clip_denoised=False,
                                            progress=False,
                                            device=device)
        else:
            batch_seed = (torch.arange(batch_size) + seed).to(device)
            samples = model.sample(model_kwargs, )

        # Append the generated latents for the following generation
        samples = samples.clip_(0, 1)


        ## Unpack and decode
        # level 0
        sample_ = torch.zeros(batch_size, length // 64, dataset.get_level_vec_len(0)).to(device)
        sample_[:, :, -7] = samples[:, :, -4].reshape(batch_size, length // 64, 64).mean(dim=-1)
        sample_[:, :, -3:] = samples[:, :, -3:].reshape(batch_size, length // 64, 64, 3).mean(dim=-2)
        decoded.append(sample_.clone())
        # level 2
        indices = samples[:, :, :3 ** 3]
        grid_values = vae_model_list[0].decode_code(indices)
        x2 = torch.cat([grid_values, samples[:, :, 3 ** 3:3 ** 3 + 14]], dim=-1)
        decoded.append(x2)

        # Denormalize and dump
        for j in range(batch_size):
            x0 = dataset.denormalize(decoded[0][j], 0).detach().cpu()
            x1 = dataset.denormalize(torch.zeros((dataset.octree_root_num * 8,
                                                  dataset.get_level_vec_len(1)),
                                                  dtype=torch.float32, device=device),
                                    1).detach().cpu()
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
    args = parser.parse_args()
    main(args)
