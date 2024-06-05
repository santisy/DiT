# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from time import time
import math
import argparse
import logging
import os
import shutil
import json
import gc

from models import DiT
from bpregen_model import PlainModel

from diffusion import create_diffusion
from transport import create_transport
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from modules.edm import EDMPrecond, EDMLoss

from data.ofalg_dataset import OFLAGDataset
from utils.copy import copy_back_fn

import psutil

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

def noise_conditioning(x_list, a_list, sampler, fm_flag=False):
    x_out = []
    for x, a in zip(x_list, a_list):
        if not fm_flag:
            x_out.append(sampler.q_sample(x, a))
        else:
            _, x0, x = sampler.sample(x)
            t = 1 - a.float() / 1000.0
            x_out.append(sampler.path_sampler.plan(t, x0, x))
    return x_out

# Parameters for the learning rate schedule
warmup_steps = 5000      # Number of steps to warm up
total_steps = 200000      # Total number of training steps
final_lr_factor = 0.2    # Final learning rate is 0.1 * LR_{target}

# Lambda function to adjust learning rate
def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        # Linearly increase the learning rate during the warmup phase
        return float(current_step) / float(max(1, warmup_steps))
    # Linearly decrease the learning rate to `final_lr_factor` * LR_{target} after the warmup phase
    progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return final_lr_factor + (1 - final_lr_factor) * (1 - progress)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def get_total_memory_usage(pid):
    """Get total memory usage of a process and its children."""
    process = psutil.Process(pid)
    total_memory = process.memory_info().rss
    for child in process.children(recursive=True):
        total_memory += child.memory_info().rss
    return total_memory

def main(args):
    """
    Trains a new DiT model.
    """
    parent_pid = psutil.Process().pid

    if args.work_on_tmp_dir:
        tmp_dir = os.getenv("SLURM_TMPDIR", "")
    else:
        tmp_dir = ""

    # Which level of tree now is training?
    level_num = args.level_num

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

    map_fn = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage
    # Resume
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=map_fn)
        print(f"\033[92mResume from checkpoint {args.resume}.\033[00m")
    else:
        resume_ckpt = None

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
        sample_dir = os.path.join(run_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        # Backup the yaml config file
        shutil.copy(args.config_file, local_dir)
        # Dump the runtime args
        with open(os.path.join(local_dir, "arg_config.json"), "w") as f:
            f.write(json.dumps(vars(args), indent=2))
    else:
        logger = create_logger(None)

    # Create dataset
    dataset = OFLAGDataset(args.data_root, **config.data)
    in_ch = dataset.get_level_vec_len(1)
    m = int(math.floor(math.pow(in_ch, 1 / 3.0)))

    # Arch variables
    num_heads = config.model.num_heads
    depth = config.model.depth
    if isinstance(depth, (list, tuple)):
        depth = depth[level_num]
    hidden_size = config.model.hidden_sizes[level_num]
    sibling_num = config.model.get("sibling_num", 2)
    if isinstance(sibling_num, (list, tuple)):
        sibling_num = sibling_num[level_num]
    learn_sigma = config.diffusion.get("learn_sigma", True)

    # Other training variantions
    edm_flag = config.model.get("use_EDM", False)
    ag_flag = config.model.get("ag_flag", False)
    fm_flag = config.model.get("fm_flag", False) # Flow matching flag

    if level_num == 2:
        in_ch = int(m ** 3)
        if ag_flag:
            in_ch = int(dataset.get_level_vec_len(1))
    elif level_num == 1: # Leaf 
        # Length 14: orientation 8 + scales 3 + relative positions 3
        in_ch = int(dataset.get_level_vec_len(1) - m ** 3)
    elif level_num == 0: # Root positions and scales
        in_ch = 4

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
        condition_node_num=dataset.get_condition_num(level_num),
        condition_node_dim=dataset.get_condition_dim(level_num, sibling_num),
        # Network itself related
        hidden_size=hidden_size, # 4 times rule
        mlp_ratio=config.model.mlp_ratio,
        depth=depth,
        num_heads=num_heads,
        cross_layers=config.model.cross_layers if level_num != 0 else [],
        learn_sigma=learn_sigma,
        # Other flags
        add_inject=config.model.add_inject,
        aligned_gen=config.model.get("align_gen", [False, True, True])[level_num],
        pos_embedding_version=config.model.get("pos_emedding_version", "v1"),
        level_num=level_num,
        sibling_num=sibling_num
    ).to(device)

    if fm_flag:
        transport = create_transport()
    elif edm_flag:
        print("\033[92mUse EDM.\033[00m")
        model = EDMPrecond(model, n_latents=dataset.octree_root_num * 8 ** 2, channels=in_ch)
        edm_loss = EDMLoss()
    else:
        diffusion = create_diffusion(timestep_respacing="", **config.diffusion)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
    requires_grad(ema, False)
    model = DDP(model, device_ids=[device])
    logger.info(f"Diffusion model created.")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0)
    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["opt"])
    if not args.no_lr_decay:
        scheduler = LambdaLR(opt, lr_lambda)
    if not args.no_mixed_pr:
        scaler = GradScaler()


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
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x0_raw, x1_raw, p0, _, y in loader:

            x0 = torch.cat([x0_raw[:, :, -7].unsqueeze(dim=-1), x0_raw[:, :, -3:]], dim=-1).detach().clone().to(device)
            x1 = x1_raw[:, :, m ** 3:].detach().clone().to(device)
            if not ag_flag:
                x2 = x1_raw[:, :, :m ** 3].detach().clone().to(device)
            else:
                x2 = x1_raw.detach().clone().to(device)

            y = y.to(device)

            # According to the level_num set the training target x and the conditions
            if level_num == 0:
                x = x0 
                a = []
                xc = []
                positions = []
            
            if level_num == 1:
                x = x1
                xc = [x0,]
                a = [torch.randint(0, diffusion.num_timesteps // 5, (x.shape[0],), device=device),]
                positions = [None,]
            elif level_num == 2:
                x = x2
                B, L, C = x1.shape
                x1 = x1.reshape(B, L // sibling_num, -1)
                xc = [x0, x1]
                a = [torch.randint(0, diffusion.num_timesteps // 5, (x.shape[0],), device=device),
                     torch.randint(0, diffusion.num_timesteps // 5, (x.shape[0],), device=device)
                    ]
                positions = [None, None]

            # Noise augmentation
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(a=a, y=y, x0=xc, positions=positions)

            if fm_flag:
                xc = noise_conditioning(xc, a, transport, fm_flag=True)
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
            elif edm_flag:
                loss = edm_loss(model, x, model_kwargs=model_kwargs)
            else:
                xc = noise_conditioning(xc, a, diffusion)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                with autocast(enabled=not args.no_mixed_pr):
                    loss_dict = diffusion.training_losses(model, x, t,
                                                          model_kwargs=model_kwargs)
                loss = loss_dict["loss"].mean()

            # Gradient step and more
            opt.zero_grad()
            if not args.no_mixed_pr:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            update_ema(ema, model.module)

            if not args.no_lr_decay:
                # Learning rate scheduler
                scheduler.step()

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
                if  not args.no_lr_decay:
                    log_info += f", Learning Rate: {scheduler.get_lr()}"

                total_memory = get_total_memory_usage(parent_pid)
                log_info += f", Total Memory Usage: {total_memory / (1024 * 1024)} MB"

                logger.info(log_info)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                # Manual gc
                gc.collect()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = os.path.join(checkpoint_dir, f"{train_steps:07d}_l{level_num}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if args.work_on_tmp_dir:
                        copy_back_fn(checkpoint_path, local_dir)
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
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
    parser.add_argument("--level-num", type=int, required=True)
    parser.add_argument("--work-on-tmp-dir", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-mixed-pr", action="store_true")

    args = parser.parse_args()
    main(args)
