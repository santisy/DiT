# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import torch.nn as nn
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from ruamel.yaml import YAML
from easydict import EasyDict as edict
from models import DiT
import numpy as np

from vae_model import VAE
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
                    torch.from_numpy(vae_std_loaded["std0"]).unsqueeze(dim=0).unsqueeze(dim=0).clone().to(device),
                    torch.from_numpy(vae_std_loaded["std1"]).unsqueeze(dim=0).unsqueeze(dim=0).clone().to(device),
                    torch.from_numpy(vae_std_loaded["std2"]).unsqueeze(dim=0).unsqueeze(dim=0).clone().to(device),
                    ]

    for l in range(3):
        in_ch = dataset.get_level_vec_len(l)
        hidden_size = int(in_ch * 8)
        latent_dim = in_ch // config.vae.latent_ratio
        vae_model = VAE(config.vae.layer_num,
                        in_ch,
                        hidden_size,
                        latent_dim)
        vae_model.load_state_dict(vae_ckpt["model"][l])
        vae_model = vae_model.to(device)
        vae_model_list.append(vae_model)
    vae_model_list.eval()


    # Model
    model_list = []
    for l in range(3):
        in_ch = dataset.get_level_vec_len(l)
        in_ch = in_ch // config.vae.latent_ratio # This is for VAE
        num_heads = config.model.num_heads
        hidden_size = int(np.ceil((in_ch * 4) / float(num_heads)) * num_heads)
        depth = config.model.depth
        if l == 1 or l == 2:
            hidden_size = hidden_size * 8
        condition_node_dim = [dim // config.vae.latent_ratio for dim in dataset.get_condition_dim(l)]

        # Create model:
        model = DiT(
            # Data related
            in_channels=in_ch, # Combine to each children
            num_classes=config.data.num_classes,
            condition_node_num=dataset.get_condition_num(l),
            condition_node_dim=condition_node_dim,
            # Network itself related
            hidden_size=hidden_size, # 4 times rule
            mlp_ratio=config.model.mlp_ratio,
            depth=depth,
            num_heads=config.model.num_heads,
            # Other flags
            add_inject=config.model.add_inject,
            aligned_gen=True if l != 0 else False
        ).to(device)
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
            length = int(dataset.octree_root_num * 8 ** l)
            z = torch.randn(batch_size,
                            length, 
                            dataset.get_level_vec_len(l) // config.vae.latent_ratio).to(device)
            a = [torch.zeros(batch_size).to(device) for _ in range(l)]
            model_kwargs = dict(a=a, y=None, x0=xc, positions=positions)

            # Sample images:
            samples = diffusion.p_sample_loop(model_list[l].forward,
                                              z.shape,
                                              z,
                                              model_kwargs=model_kwargs,
                                              clip_denoised=False,
                                              progress=False,
                                              device=device)

            # Append the generated latents for the following generation
            xc.append(samples.clone())

            # Rescale and decode
            samples = samples * vae_std_list[l]
            samples = samples.reshape(batch_size * length, -1)
            samples = vae_model_list[l].decode(samples)
            samples = samples.reshape(batch_size, length, -1)
            decoded.append(samples)

            # Get the positions and scales from generated
            if l > 0:
                scale = torch.zeros(batch_size, length).to(device)
                positions.append(load_utils.deduce_position_from_sample(scales[-1], scale, positions[-1], length))
                scales.append(scale)
            else:
                scales.append(dataset.rescale_voxel_len(samples[:, :, -7].clone()))
                positions.append(dataset.rescale_positions(samples[:, :, -3:].clone()))

        # Denormalize and dump
        for j in range(batch_size):
            x0 = dataset.denormalize(decoded[0][j], 0).detach().cpu()
            x1 = dataset.denormalize(decoded[1][j], 1).detach().cpu()
            x2 = dataset.denormalize(decoded[2][j], 2).detach().cpu()
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
