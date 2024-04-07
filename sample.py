# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
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

import argparse
from data.ofalg_dataset import OFLAGDataset
from data_extensions import load_utils


def main(args):
    # Make directories
    os.makedirs(args.export_dir, exist_ok=True)
    out_dir = args.export_dir

    # Create dataset. For denormalizing
    dataset = OFLAGDataset(args.data_root, **config.data)

    # Load config
    with open(args.config_file, "r") as f:
        yaml = YAML()
        config = edict(yaml.load(f))

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_list = []
    for l in range(3):
        in_ch = dataset.get_level_vec_len(l)
        num_heads = config.model.num_heads
        hidden_size = int(np.ceil((in_ch * 4) / float(num_heads)) * num_heads)
        depth = config.model.depth
        if l == 1:
            hidden_size = hidden_size * 3
        if l == 2:
            hidden_size = hidden_size * 2
        # Create model:
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
            num_heads=config.model.num_heads,
            # Other flags
            add_inject=config.model.add_inject,
            aligned_gen=True if l != 0 else False
        ).to(device)
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = args.ckpt[l]
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
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
        for l in range(3):
            length = int(dataset.octree_root_num * 8 ** l)
            z = torch.randn(batch_size,
                            length, 
                            dataset.get_level_vec_len(l))
            model_kwargs = dict(y=None, x0=xc, positions=positions)
            # Sample images:
            samples = diffusion.p_sample_loop(model.forward,
                                              z.shape,
                                              z,
                                              model_kwargs=model_kwargs,
                                              clip_denoised=False,
                                              progress=False,
                                              device=device)
            xc.append(samples)
            # TODO: get the positions from generated
            if l > 0:
                scale = torch.zeros(batch_size, length, 3).to(device)
                positions.append(load_utils.deduce_position_from_sample(scales[-1], scale, positions[-1], length))
                scales.append(scale)
            else:
                scales.append(samples[:, :, -7].clone() * dataset.max_voxel_len)
                positions.append(samples[:, :, -3:].clone())

        # Denormalize and dump
        for j, (x0, x1, x2) in enumerate(zip(xc)):
            x0 = dataset.denormalize(x0, 0).detach().cpu()
            x1 = dataset.denormalize(x1, 1).detach().cpu()
            x2 = dataset.denormalize(x2, 2).detach().cpu()
            load_utils.dump_to_bin(os.path.join(out_dir, f"out_{j + i * batch_size:04d}.bin"),
                                   x0, x1, x2, dataset.octree_root_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", nargs='+',
                        help="A list of strings that provides three level generation models.")
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, require=True)
    parser.add_argument("--sample-num", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args)
