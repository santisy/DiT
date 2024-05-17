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
from einops import rearrange

from vae_model import VAE, VAELinear, loss_function

from data.ofalg_dataset import OFLAGDataset
from utils.copy import copy_back_fn
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast


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


def random_sample_and_reshape(x, l, m, level_num=2, zero_ratio=None):
    if level_num == 2:
        x = x[:, :, :m**3]
        B, L, C = x.shape
        x = rearrange(x, 'b (l n1 n2 n3) (x y z) -> b l (n1 x) (n2 y) (n3 z)',
                n1=2, n2=2, n3=2, x=5, y=5, z=5)
        x = x.reshape(B, L // 8, -1).contiguous()
    else:
        x = x[:, :, m**3:]
        B, L, C = x.shape
        x = x.reshape(B, L // 8, C * 8)

    out = []
    for i in range(x.size(0)):
        indices = torch.randperm(x.size(1))[:l]
        out.append(x[i, indices, :].unsqueeze(dim=0))

    out = torch.cat(out, dim=0)
    return out.contiguous().clone()
    
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
    level_num = args.level_num
    m = None
    in_ch = dataset.get_level_vec_len(2)
    m = int(math.floor(math.pow(in_ch, 1 / 3.0)))

    model = VAE(config.vae.layer_n,
                config.vae.in_ch,
                config.vae.latent_ch,
                m * 2,
                quant_code_n=config.vae.get("quant_code_n", 2048),
                quant_version=config.vae.get("quant_version", "v0"),
                quant_heads=config.vae.get("quant_heads", 1),
                downsample_n=config.vae.get("downsample_n", 1),
                kl_flag=config.vae.get("kl_flag", False),
                level_num=level_num)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"][0])
        ema.load_state_dict(resume_ckpt["ema"][0])
    requires_grad(ema, False)
    ema_list.append(ema)
    # Put DDP on this
    model = DDP(model.to(device), device_ids=[rank])
    model_list.append(model)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model_list.parameters(), lr=0.0001)
    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["opt"])
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
    scaler = GradScaler()

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
            # Do not sample too much zero entries when training VAE
            sample_num = 256 if level_num == 1 else 128
            x2 = random_sample_and_reshape(x2.to(device), sample_num, m,
                                           level_num=level_num,
                                           zero_ratio=0.1)

            x_list = [x2,]

            loss = 0

            for _, (x, model) in enumerate(zip(x_list, model_list)):
                with autocast():
                    x_rec, q_loss, _ = model(x)
                    loss_, recon_loss = loss_function(x_rec, x, q_loss)
                loss += loss_

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            #clipped_norm = torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), max_norm=100.0)
            #print(clipped_norm)
            scaler.step(opt)
            scaler.update()

            for ema, model in zip(ema_list, model_list):
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += recon_loss.item()
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
                        if level_num == 2:
                            x2 = x2[:, :, :m**3]
                            B, L, C = x2.shape
                            x2 = rearrange(x2, 'b (l n1 n2 n3) (x y z) -> b l (n1 x) (n2 y) (n3 z)',
                                    n1=2, n2=2, n3=2, x=5, y=5, z=5)
                            x2 = x2.reshape(B, L // 8, -1).contiguous().clone()
                        else:
                            x2 = x2[:, :, m**3:]
                            B, L, C = x2.shape
                            x2 = x2.reshape(B, L // 8, C * 8).contiguous().clone()
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
    parser.add_argument("--level-num", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
