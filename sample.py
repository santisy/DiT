# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from diffusion.respace import SpacedDiffusion
from ruamel.yaml import YAML
from easydict import EasyDict as edict
from models import DiT

import argparse
from data.ofalg_dataset import OFLAGDataset
from bpregen_model import PlainModel
from data_extensions import load_utils
from torch.cuda.amp import autocast
from transport import create_transport, Sampler


def main(args):
    # Make directories
    os.makedirs(args.export_dir, exist_ok=True)
    out_dir = args.export_dir

    # Load config
    config_list = []
    for config_file in args.config_file:
        with open(config_file, "r") as f:
            yaml = YAML()
            config_list.append(edict(yaml.load(f)))

    # Create dataset. For denormalizing
    dataset = OFLAGDataset(args.data_root, only_infer=True, **config_list[0].data)

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_ch = dataset.get_level_vec_len(1)
    m = int(math.floor(math.pow(in_ch, 1 / 3.0)))

    # Model
    model_list = []
    in_ch_list = []
    sampler_list = []
    fm_flags = []
    m_ = None
    for l in range(3):
        config = config_list[l]

        ag_flag = config.model.get("ag_flag", False)
        fm_flag = config.model.get("fm_flag", False) # Flow matching flag
        fm_flags.append(fm_flag)

        sibling_total = config.model.get("sibling_num", 2)
        depth_total = config.model.depth
        learn_sigma = config.diffusion.get("learn_sigma", True)
        num_heads = config.model.num_heads

        if isinstance(depth_total, (list, tuple)):
            depth = depth_total[l]
        else:
            depth = depth_total
        if isinstance(sibling_total, (list, tuple)):
            sibling_num = sibling_total[l]
        else:
            sibling_num = sibling_total

        hidden_size = config.model.hidden_sizes[l]

        if l == 2:
            in_ch = int(m ** 3)
            if ag_flag:
                in_ch = int(dataset.get_level_vec_len(1))
            in_ch_list.append(in_ch)
        elif l == 1: # Leaf 
            # Length 14: orientation 8 + scales 3 + relative positions 3
            in_ch = int(dataset.get_level_vec_len(1) - m ** 3)
            in_ch_list.append(in_ch)
        elif l == 0: # Root positions and scales
            in_ch = 4
            in_ch_list.append(in_ch)

        if config.model.get("plain_model", False):
            model_class = PlainModel
            learn_sigma = False
        else:
            model_class = DiT

        # Create DiT model
        model = model_class(
            # Data related
            in_channels=in_ch, # Combine to each children
            num_classes=config.data.num_classes,
            condition_node_num=dataset.get_condition_num(l),
            condition_node_dim=dataset.get_condition_dim(l, sibling_num),
            # Network itself related
            hidden_size=hidden_size, # 4 times rule
            mlp_ratio=config.model.mlp_ratio,
            depth=depth,
            num_heads=num_heads,
            cross_layers=config.model.cross_layers if l != 0 else [],
            learn_sigma=learn_sigma,
            # Other flags
            add_inject=config.model.add_inject,
            aligned_gen=config.model.get("align_gen", [False, True, True])[l],
            pos_embedding_version=config.model.get("pos_emedding_version", "v1"),
            level_num=l,
            sibling_num=sibling_num
        )
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = args.ckpt[l]
        print(f"\033[92mLoading model level {l}: {ckpt_path}.\033[00m")
        model_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_ckpt["model"])
        model.to(device)
        model.eval()  # important!
        model_list.append(model)

        # Create samplers
        if fm_flag:
            transport = create_transport()
            sampler = Sampler(transport)
        else:
            sampler = create_diffusion(timestep_respacing="", **config.diffusion)
        sampler_list.append(sampler)


        if args.only_l0:
            break


    batch_size = args.sample_batch_size
    sample_num = args.sample_num
    if args.sample_all:
        sample_num = dataset.get_sample_num()
        print(f"\033[92mSample all five percent objects {sample_num}.\033[00m")
    for i in range(sample_num // batch_size + 1):
        xc = []
        positions = []
        scales = []
        decoded = []
        for l in range(3):
            # Random generator
            seed = i * 3 + l
            if args.l0_seed is not None and l == 0:
                seed = args.l0_seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            # Shape parameter
            length = dataset.octree_root_num if l == 0 else dataset.octree_root_num * 8
            ch = in_ch_list[l]

            # Random input
            z = torch.randn(batch_size,
                            length, 
                            ch,
                            generator=generator,
                            device=device)
            a = [torch.zeros((z.shape[0],) , dtype=torch.int64, device=device) for _ in range(l)]
            model_kwargs = dict(a=a, y=None, x0=xc, positions=positions)

            # Sample
            with autocast():
                if fm_flags[l]:
                    sampler: Sampler = sampler_list[l]
                    sample_fn = sampler.sample_ode(
                                sampling_method="euler",
                                num_steps=60,
                                atol=1e-6,
                                rtol=1e-3,
                                reverse=False,
                                time_shifting_factor=4,
                    )
                    samples = sample_fn(z, model_list[l].forward, **model_kwargs)[-1]
                else:
                    sampler: SpacedDiffusion = sampler_list[l]
                    samples = sampler.p_sample_loop(model_list[l].forward,
                                                    z.shape,
                                                    z,
                                                    model_kwargs=model_kwargs,
                                                    clip_denoised=False,
                                                    progress=False,
                                                    device=device)

            # Append the generated latents for the following generation
            samples = samples.float().clip_(0, 1)

            if l == 1:
                B, L, C = samples.shape
                samples = samples.reshape(B, L // sibling_num, -1).contiguous()
            xc.append(samples.clone())

            if l == 2:
                # Rescale and decode
                B, L, C = xc[-2].shape
                x2_non_V = xc[-2].reshape(B, L * sibling_num, -1).clone()
                decoded.append(torch.cat([samples, x2_non_V], dim=-1).clone())
            elif l == 0:
                sample_ = torch.zeros(batch_size,
                                      length,
                                      dataset.get_level_vec_len(0) - 4).to(device)
                sample_[:, :, -7] = samples[:, :, 0]
                sample_[:, :, -3:] = samples[:, :, -3:]
                decoded.append(sample_.clone())

            # Get the positions and scales from generated
            scales.append(None)
            positions.append(None)

            if args.only_l0:
                break

        # Denormalize and dump
        for j in range(batch_size):
            x0 = dataset.denormalize(decoded[0][j], 0).detach().cpu()
            if not args.only_l0:
                x1 = dataset.denormalize(decoded[1][j], 1).detach().cpu()
            else:
                x1 = torch.zeros(dataset.octree_root_num * 8,
                                 dataset.get_level_vec_len(1) - 4)
                x1 = dataset.denormalize(x1, 1)
            load_utils.dump_to_bin(os.path.join(out_dir, f"out_{j + i * batch_size:04d}.bin"),
                                   x0, x1, dataset.octree_root_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", nargs='+',
                        help="A list of strings that provides three level generation models.")
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--config-file", nargs='+',
                        help="List of config files")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("-a", "--sample_all", action="store_true",
                        help="Sample all the needed samples to calculate the metrics.")
    parser.add_argument("--sample-num", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--l0_seed", type=int, default=None,
                        help="Given and fixed l0 random seed.")
    parser.add_argument("--only-l0", action="store_true",
                        help="Only inference the l1")
    args = parser.parse_args()
    main(args)
