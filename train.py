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
import numpy as np
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

from models import DiT
from vae_model import VAE, VAELinear
from diffusion import create_diffusion
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast
from edm import EDMPrecond, EDMLoss

from data.ofalg_dataset import OFLAGDataset
from utils.copy import copy_back_fn


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

def noise_conditioning(x_list, a_list, diff):
    x_out = []
    for x, a in zip(x_list, a_list):
        x_out.append(diff.q_sample(x, a))
    return x_out

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
    in_ch = dataset.get_level_vec_len(2)
    m = int(math.floor(math.pow(in_ch, 1 / 3.0)))

    # Prepare VAE model
    if level_num in [1, 2]:
        vae_model_list = nn.ModuleList()
        for l in range(1, 3):
            vae_ckpt = torch.load(args.vae_ckpt[l - 1], map_location=map_fn)
            vae_model = VAE(config.vae.layer_n,
                            config.vae.in_ch,
                            config.vae.latent_ch,
                            m * 2,
                            quant_code_n=config.vae.get("quant_code_n", 2048),
                            quant_version=config.vae.get("quant_version", "v0"),
                            quant_heads=config.vae.get("quant_heads", 1),
                            downsample_n=config.vae.get("downsample_n", 1),
                            level_num=level_num)

            # Load the trained model
            vae_model.load_state_dict(vae_ckpt["model"])
            vae_model = vae_model.to(device)
            vae_model_list.append(vae_model)
        vae_model_list.eval()
        logger.info("VAE model created and loaded.")

    # Arch variables
    num_heads = config.model.num_heads
    depth = config.model.depth
    hidden_size = config.model.hidden_sizes[level_num]
    edm_flag = config.model.get("use_EDM", False)

    if level_num == 2:
        in_ch = dataset.get_level_vec_len(2)
        in_ch = int(math.ceil(m / 2) ** 3) * 2
    elif level_num == 1: # Leaf 
        # Length 14: orientation 8 + scales 3 + relative positions 3
        in_ch = int(dataset.get_level_vec_len(2) - m ** 3) * 8 // 4
    elif level_num == 0: # Root positions and scales
        in_ch = 4

    # Create DiT model
    model = DiT(
        # Data related
        in_channels=in_ch, # Combine to each children
        num_classes=config.data.num_classes,
        condition_node_num=dataset.get_condition_num(level_num),
        condition_node_dim=dataset.get_condition_dim(level_num),
        # Network itself related
        hidden_size=hidden_size, # 4 times rule
        mlp_ratio=config.model.mlp_ratio,
        depth=depth,
        num_heads=num_heads,
        cross_layers=config.model.cross_layers if level_num != 0 else [],
        learn_sigma=config.diffusion.get("learn_sigma", True),
        # Other flags
        add_inject=config.model.add_inject,
        aligned_gen=False,
        pos_embedding_version=config.model.get("pos_emedding_version", "v1"),
        level_num=level_num
    ).to(device)
    if edm_flag:
        print("\033[92mUse EDM.\033[00m")
        model = EDMPrecond(model, n_latents=dataset.octree_root_num * 8 ** 2, channels=in_ch)
        edm_loss = EDMLoss()
    else:
        diffusion = create_diffusion(timestep_respacing="",
                                    **config.diffusion)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
    requires_grad(ema, False)
    model = DDP(model, device_ids=[rank])
    logger.info(f"Diffusion model created.")

    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    #logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0)
    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["opt"])
    if not args.no_lr_decay:
        scheduler = StepLR(opt, step_size=5, gamma=0.999)


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
        for x0, _, x2, p0, _, p2, y in loader:

            # To device, encode VAE and divide the per-element statistics
            x0 = torch.cat([x0[:, :, -7].unsqueeze(dim=-1), x0[:, :, -3:]], dim=-1).to(device)
            x1 = x2[:, :, m ** 3:].clone().to(device)
            B, L, C = x1.shape 
            x1 = x1.reshape(B, L // 8, C * 8).contiguous()
            x2 = x2[:, :, :m ** 3].clone().to(device)
            B, L, C = x2.shape
            x2 = x2.reshape(B, L // 8, C * 8)

            # VAE auto-encoding
            if level_num >= 1:
                with torch.no_grad():
                    x1_list = []
                    for x1_ in torch.split(x1, 4, dim=0):
                        with autocast():
                            x1_list.append(vae_model_list[0].get_normalized_indices(x1_))
                    x1 = torch.cat(x1_list, dim=0).detach().float()
            elif level_num >= 2:
                with torch.no_grad():
                    x2_list = []
                    for x2_ in torch.split(x2, 4, dim=0):
                        with autocast():
                            x2_list.append(vae_model_list[1].get_normalized_indices(x2_))
                    x2 = torch.cat(x2_list, dim=0).detach().float()


            p2 = p2.to(device)
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
                a = [torch.randint(0, diffusion.num_timesteps // 2, (x.shape[0],), device=device),]
                positions = [None,]
            elif level_num == 2:
                x = x2
                xc = [x0, x1]
                a = [torch.randint(0, diffusion.num_timesteps // 2, (x.shape[0],), device=device),
                     torch.randint(0, diffusion.num_timesteps // 2, (x.shape[0],), device=device)
                    ]
                positions = [None, None]

            # Noise augmentation
            xc = noise_conditioning(xc, a, diffusion)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(a=a, y=y, x0=xc, positions=positions)
            if not edm_flag:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t,
                                                      model_kwargs=model_kwargs)
                loss = loss_dict["loss"].mean()
            else:
                loss = edm_loss(model, x, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
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
                if not args.no_lr_decay:
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

        if not args.no_lr_decay:
            scheduler.step()

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

    # VAE related
    parser.add_argument("--vae-ckpt", nargs="+", help="VAE checkpoints")

    args = parser.parse_args()
    main(args)
