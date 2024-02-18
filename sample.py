# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from ruamel.yaml import YAML
from easydict import EasyDict as edict
from models import DiT
import numpy as np
import cv2
from utils.tree_to_img import tree_to_img_mnist

import argparse


def main(args):

    # Load config
    with open(args.config_file, "r") as f:
        yaml = YAML()
        config = edict(yaml.load(f))

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model:
    model = DiT(
        # Data related
        in_channels=config.data.in_channels,
        num_classes=config.data.num_classes,
        condition_node_num=config.data.condition_node_num,
        condition_node_dim=config.data.condition_node_dim,
        # Network itself related
        hidden_size=config.model.hidden_size,
        mlp_ratio=config.model.mlp_ratio,
        depth=config.model.depth,
        num_heads=config.model.num_heads
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(timestep_respacing="",
                                 **config.diffusion)

    # Create sampling noise:
    n = 4
    z = torch.randn(4, 64, 3, device=device)
    y = torch.tensor([-1,] * n, device=device)

    # Get previous layer condition
    from data.permutedDataset import MNISTPermutedDataset
    dataset = MNISTPermutedDataset()
    x0 = []
    for i in range(n):
        data0, _, _ = dataset[i]
        x0.append(torch.tensor(data0, device=device))
    x0 = torch.stack(x0)

    model_kwargs = dict(y=y, x0=x0)

    # Sample images:
    samples = diffusion.p_sample_loop(model.forward,
                                      z.shape,
                                      z,
                                      model_kwargs=model_kwargs,
                                      clip_denoised=False,
                                      progress=True,
                                      device=device)

    # Save and display images:
    data0 = ((x0 + 1.0) / 2.0).detach().cpu().numpy()
    data0 = np.clip(data0, 0.0, 1.0)
    data1 = ((samples + 1.0) / 2.0).detach().cpu().numpy()
    data1 = np.clip(data1, 0.0, 1.0)
    for i in range(4): # Sample up to 4 images
        img0, img1 = tree_to_img_mnist(data0[i].flatten(),
                                       data1[i].flatten())
        cv2.imwrite(f"test_imgs/samples{i}_0.png", img0)
        cv2.imwrite(f"test_imgs/smaples{i}_1.png", img1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
