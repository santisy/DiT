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

import argparse
from data.ofalg_dataset import OFLAGDataset
from data_extensions import load_utils
from memory_profiler import profile

@profile
def main(args):
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

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        prefetch_factor=2,
                        num_workers=12)

    for i, (x0, x1, _, _, _) in enumerate(tqdm(loader)):
        x0 = x0.to(device)
        x1 = x1.to(device)
        if i > 100:
            break
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args)