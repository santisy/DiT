# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import math
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from time import time
import argparse
import logging
import os
import shutil
import json

from vae_model import VAE, VAELinear, loss_function

from data.ofalg_dataset import OFLAGDataset
from utils.copy import copy_back_fn
from torch.optim.lr_scheduler import StepLR


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def random_sample_and_reshape(x, l, m, zero_ratio=None):
    out = []
    for i in range(x.size(0)):
        if zero_ratio is None:
            indices = torch.randperm(x.size(1))[:l]
            out.append(x[i, indices, :])
        else:
            assert zero_ratio < 1.0
            zero_num = int(l * zero_ratio)
            non_zero_num = l - zero_num
            zero_indices = torch.where(x[i, :, -14:-6].sum(dim=1) == 4.0)[0]
            non_zero_indices = torch.where(x[i, :, -14:-6].sum(dim=1) != 4.0)[0]
            out.append(x[i, zero_indices[torch.randperm(zero_indices.numel())[:zero_num]], :])
            out.append(x[i, non_zero_indices[torch.randperm(non_zero_indices.numel())[:non_zero_num]], :])

    out = torch.cat(out, dim=0).unsqueeze(dim=0)
    return out[:, :, :m ** 3].clone()
    
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    if args.work_on_tmp_dir:
        tmp_dir = os.getenv("SLURM_TMPDIR", "")
    else:
        tmp_dir = ""

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Load config
    with open(args.config_file, "r") as f:
        yaml = YAML()
        config = edict(yaml.load(f))

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        local_dir = os.path.join(args.results_dir, args.exp_id)
        os.makedirs(local_dir, exist_ok=True)
        run_dir = os.path.join(tmp_dir, args.results_dir, args.exp_id)
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(run_dir)
        logger.info(f"Experiment directory created at {run_dir}")
        val_record_file = os.path.join(run_dir, "val_record.txt")
        val_record = open(val_record_file, "w")
        # Backup the yaml config file
        shutil.copy(args.config_file, local_dir)
        # Dump the runtime args
        with open(os.path.join(local_dir, "arg_config.json"), "w") as f:
            f.write(json.dumps(vars(args), indent=2))
    else:
        logger = create_logger(None)

    # Create dataset
    dataset = OFLAGDataset(args.data_root, validate_num=10, **config.data)
    val_dataset = OFLAGDataset(args.data_root, validate_num=10, validate_flag=True,
                               **config.data)

    # Temp variables
    model_list = nn.ModuleList()
    ema_list = nn.ModuleList()
    linear_flag = config.vae.linear

    for l in range(2, 3):
        in_ch = dataset.get_level_vec_len(l)
        m = int(math.floor(math.pow(in_ch, 1 / 3.0)))
        if not linear_flag:
            print("\033[92mUse conv VAE.\033[00m")
            model = VAE(config.vae.layer_n,
                        config.vae.in_ch,
                        config.vae.latent_ch,
                        m)
        else:
            in_ch = int(m ** 3)
            latent_dim = in_ch // config.vae.latent_ratio
            print("\033[92mUse linear VAE.\033[00m")
            model = VAELinear(config.vae.layer_n,
                              in_ch,
                              in_ch * 16,
                              latent_dim)
        
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        ema_list.append(ema)

        # Put DDP on this
        model = DDP(model.to(device), device_ids=[rank])
        model_list.append(model)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model_list.parameters(), lr=0.0002)
    scheduler = StepLR(opt, step_size=3, gamma=0.999)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=int(args.num_workers // dist.get_world_size()),
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):}")

    # Prepare models for training:
    for ema, model in zip(ema_list, model_list):
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model_list.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_list.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for _, _, x2, _, _, _, _ in loader:

            # To device
            #x0 = random_sample_and_reshape(x0.to(device), 64)
            #x1 = random_sample_and_reshape(x1.to(device), 256)
            # Do not sample too much zero entries when training VAE
            if not linear_flag:
                x2 = random_sample_and_reshape(x2.to(device), 400, m, zero_ratio=0.1)
            else:
                x2 = random_sample_and_reshape(x2.to(device), 1024, m, zero_ratio=0.1)
            x_list = [x2,]

            loss = 0

            for _, (x, model) in enumerate(zip(x_list, model_list)):
                x_rec, mean, logvar = model(x)
                loss += loss_function(x_rec, x, mean, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            for ema, model in zip(ema_list, model_list):
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                log_info = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                log_info += f", Learning Rate: {scheduler.get_lr()}"
                logger.info(log_info)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": [model.module.state_dict() for model in model_list],
                        "ema": [ema.state_dict() for ema in ema_list],
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = os.path.join(checkpoint_dir, f"vae_{train_steps:07d}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    # Calculate validation error
                    val_loss = 0
                    for i in range(len(val_dataset)):
                        _, _, x2, _, _, _, _ = val_dataset[i]
                        x2 = x2.unsqueeze(dim=0).to(device)
                        x2 = x2[:, :, :m ** 3]
                        with torch.no_grad():
                            x2_rec, _, _ = ema_list[0](x2)
                            val_loss += (x2_rec - x2).abs().mean()
                    val_loss = val_loss / len(val_dataset)
                    val_record.write(f"Step {train_steps}:\t{val_loss.cpu().item():.4f}\n")
                    val_record.flush()
                    if args.work_on_tmp_dir:
                        copy_back_fn(checkpoint_path, local_dir)
                        copy_back_fn(val_record_file, local_dir)

                dist.barrier()

        scheduler.step()

    model_list.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    val_record.close()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="training_runs")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--no-lr-decay", action="store_true")

    # Newly added argument
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--work-on-tmp-dir", action="store_true")

    args = parser.parse_args()
    main(args)
