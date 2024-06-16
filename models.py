# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
from typing import List

from timm.models.vision_transformer import Mlp
try:
    from flash_attn import flash_attn_func as sdp_atten_fn
    print("\033[92m Use flash attention.\033[00m")
except:
    from torch.nn.functional import scaled_dot_product_attention as sdp_atten_fn
    print("\033[92m Use pytorch attention.\033[00m")

from utils.positional_embedding import fourier_positional_encoding

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = sdp_atten_fn(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps, Class Labels, and Previous Node #
#################################################################################

class PreviousNodeEmbedder(nn.Module):
    def __init__(self,
                 node_dim: List[int],
                 hidden_size: int,
                 PEV="v1",
                 level_num=0):
        super().__init__()

        self.mlp_list = nn.ModuleList()
        self.PEV = PEV # Positional embedding version

        for i, nd in enumerate(node_dim):
            self.mlp_list.append(
                nn.Sequential(
                nn.Linear(nd, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                )
            )

    def forward(self, nodes: List[torch.Tensor], PEs: List[torch.Tensor]):
        embedded_out = []
        for i, (n, pe, mlp) in enumerate(zip(nodes, PEs, self.mlp_list)):
            if self.PEV != "v2":
                out = mlp(n) + pe
            else:
                out = mlp(n)
            embedded_out.append(out)
        return embedded_out


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class CrossAttention(nn.Module):
    """
        Cross Attention Layer
    """
    def __init__(self, hidden_size, num_heads, PEV="v1"):
        super().__init__()
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_back = nn.Linear(hidden_size, hidden_size, bias=True)
        self.PEV = PEV

        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                                         num_heads=num_heads,
                                                         batch_first=True)

    def forward(self, x, c):
        if self.PEV == "v2":
            batch_size = x.size(0)
            seq_x = x.size(1)
            seq_c = c.size(1)
            if seq_x == seq_c:
                attn_mask = -10 * torch.ones(seq_x, seq_x)
                attn_mask.fill_diagonal_(0)
                attn_mask = attn_mask.to(x.device)
            else:
                attn_mask = None
        else:
            attn_mask = None

        q = self.proj_q(x)
        k = self.proj_k(c)
        v = self.proj_v(c)

        attn_out, _ = self.multihead_attention(q, k, v, attn_mask=attn_mask)
        attn_out = self.proj_back(attn_out)

        return attn_out


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads,
                 mlp_ratio=4.0, 
                 cond_num=1,
                 cross_attention=False,
                 add_inject=False,
                 PEV="v1",
                 real_noa=False,
                 **block_kwargs):
        super().__init__()
        self.cross_attention = cross_attention
        self.add_inject = add_inject
        self.cond_num = cond_num
        self.real_noa = real_noa
        self.PEV = PEV

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size,
                              num_heads=num_heads,
                              qkv_bias=True,
                              attn_drop=0.1,
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        if cross_attention: # This is a flag meant for conditional injection
            if not add_inject:
                self.cross_list = nn.ModuleList()
                self.norm0_list = nn.ModuleList()
                self.norm00_list = nn.ModuleList()
                self.adaLN_modulation_mca_list = nn.ModuleList()

                for _ in range(cond_num):
                    self.cross_list.append(CrossAttention(hidden_size, num_heads, PEV=PEV))
                    self.norm0_list.append(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
                    self.norm00_list.append(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
                    self.adaLN_modulation_mca_list.append(nn.Sequential(nn.SiLU(),
                                                          nn.Linear(hidden_size, 5 * hidden_size, bias=True)))
            else:
                self.adaLN_modulation_mca_list = nn.ModuleList()
                self.norm0_list = nn.ModuleList()
                self.mlp_list = nn.ModuleList()
                for _ in range(cond_num):
                    self.norm0_list.append(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
                    if not real_noa:
                        self.adaLN_modulation_mca_list.append(nn.Sequential(nn.SiLU(),
                                                            nn.Linear(hidden_size, 3 * hidden_size, bias=True)))
                    self.mlp_list.append(nn.Linear(hidden_size, hidden_size))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, x, c, x0, a_list):
        # x0 means the previous level embedded results (condition)

        if self.cross_attention:
            if not self.add_inject:
                for cross, norm0, norm00, adaMM, x0_ in zip(self.cross_list, self.norm0_list, self.norm00_list, self.adaLN_modulation_mca_list, x0):
                    gate_mca, shift_mca, scale_mca, shift_mca0, scale_mca0 = adaMM(c).chunk(5, dim=1)
                    x = x + gate_mca.unsqueeze(1) * cross(modulate(norm0(x), shift_mca, scale_mca),
                                                          modulate(norm00(x0_), shift_mca0, scale_mca0))
            else:
                for mlp_, x0_, a, norm0, adaMM in zip(self.mlp_list, x0, a_list, self.norm0_list, self.adaLN_modulation_mca_list):
                    seq_0 = x0_.size(1)
                    seq_x = x.size(1)
                    if seq_0 == seq_x:
                        add_ = x0_
                    else:
                        add_ = torch.repeat_interleave(x0_, seq_x // seq_0, dim=1)
                    if not self.real_noa:
                        gate_mca, shift_mca, scale_mca = adaMM(a).chunk(3, dim=1)
                        x = x + gate_mca.unsqueeze(1) * mlp_(modulate(norm0(add_), shift_mca, scale_mca))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class InputLayer(nn.Module):
    """
    The input layer of DiT.
    """
    def __init__(self, hidden_size, in_channels):
        super().__init__()
        self.norm_input = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_channels, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.linear(x)
        x = modulate(self.norm_input(x), shift, scale)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels,
                 sibling_num=4,
                 aligned_gen=False):
        super().__init__()
        self.aligned_gen = aligned_gen
        self.sibling_num = sibling_num

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        out_ch = out_channels if not aligned_gen else out_channels * sibling_num
        self.linear = nn.Linear(hidden_size, out_ch, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        if self.aligned_gen:
            # Unpack
            B, L, C = x.shape
            x = x.reshape(B, L, self.sibling_num, C // self.sibling_num)
            x = x.reshape(B, L * self.sibling_num, C // self.sibling_num)
        return x

class PackLayer(nn.Module):
    """Packing node"""
    def __init__(self, in_channels, hidden_size, sibling_num=4):
        super().__init__()
        self.sibling_num = sibling_num
        self.map = nn.Linear(in_channels * sibling_num, hidden_size)

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, L // self.sibling_num, self.sibling_num, C)
        x = x.reshape(B, L // self.sibling_num, self.sibling_num * C)
        return self.map(x)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=4, # input token dimension size
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        # Newly added
        condition_node_num=[], # What is the number of nodes in previous level
        condition_node_dim=[],
        cross_layers=[4, 8, 12],
        add_inject=False,
        # ---- optional
        num_classes=1000,
        aligned_gen=False,
        pos_embedding_version="v1",
        level_num=0,
        sibling_num=8,
        learned_pos_embedding=False,
        real_noa=False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.sibling_num = sibling_num # How many to pack
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.condition_node_num = condition_node_num
        self.add_inject = add_inject
        self.aligned_gen = aligned_gen
        self.pos_embedding_version = pos_embedding_version
        self.level_num = level_num
        self.learned_pos_embedding = learned_pos_embedding
        self.real_noa = real_noa

        if learned_pos_embedding and level_num == 0:
            self.learned_PE = nn.Embedding(256 // sibling_num, hidden_size)
            

        # Input layer
        if not aligned_gen:
            self.input_layer = nn.Linear(in_channels, hidden_size)
        else:
            self.input_layer = PackLayer(in_channels, hidden_size, sibling_num)

        self.n_embedder = PreviousNodeEmbedder(condition_node_dim,
                                               hidden_size,
                                               PEV=pos_embedding_version,
                                               level_num=level_num)
        # This is to patchify images from NCHW -> NLC
        # We will not use in our fully tokenized transformer case
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Augmentation level embedder
        self.a_embedder = nn.ModuleList()
        for _ in range(len(condition_node_dim)):
            if not real_noa:
                self.a_embedder.append(TimestepEmbedder(hidden_size))
            else:
                self.a_embedder.append(nn.Identity())
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = None


        self.blocks = nn.ModuleList()
        block_class = partial(DiTBlock,
                              hidden_size,
                              num_heads,
                              mlp_ratio=mlp_ratio,
                              cond_num=len(condition_node_dim),
                              add_inject=add_inject,
                              PEV=self.pos_embedding_version,
                              real_noa=real_noa
                              )
        for i in range(depth):
            if i not in cross_layers:
                self.blocks.append(block_class())
            else:
                self.blocks.append(block_class(cross_attention=True))

        # We do not need final layers
        self.output_layer = FinalLayer(hidden_size, self.out_channels,
                                       sibling_num=sibling_num,
                                       aligned_gen=aligned_gen)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        #w = self.x_embedder.proj.weight.data
        #nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize input and output layer
        # # Input
        # nn.init.normal_(self.input_layer.linear.weight, std=0.02)
        # nn.init.constant_(self.input_layer.adaLN_modulation[-1].weight, 0) 
        # nn.init.constant_(self.input_layer.adaLN_modulation[-1].bias, 0) 
        # Output
        nn.init.constant_(self.output_layer.linear.weight, 0.0)
        nn.init.constant_(self.output_layer.linear.bias, 0.0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0) 
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0) 

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[3].weight, std=0.02)
        for a_embedder in self.a_embedder:
            if not self.real_noa:
                nn.init.normal_(a_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(a_embedder.mlp[3].weight, std=0.02)

        # IntiaLize node embedder
        for mlp in self.n_embedder.mlp_list:
            nn.init.normal_(mlp[0].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.cross_attention and not self.add_inject:
                for adaMM in block.adaLN_modulation_mca_list:
                    nn.init.constant_(adaMM[-1].weight, 0)
                    nn.init.constant_(adaMM[-1].bias, 0)

    def forward(self, x, t, a, y, x0, positions):
        """
            Forward pass of DiT.
            x: (N, L, C) tensor of spatial inputs (images or latent representations of images)
            t: (N,) tensor of diffusion timesteps
            a: **List** of (N,) augmentation level embedder
            y: (N,) tensor of class labels
            x0: **LIST** of (N, L0, C0) previous levels of nodes
            positions: **LIST** of (N, L0, 3) positional embeddings
        """

        # Variance preserving noising x0, and positions
        batch_size, L, C = x.shape
        PEs = []
        for i, (a_, p_) in enumerate(zip(a, positions)):
            #a_ = a_.reshape([batch_size, 1, 1])
            #x0[i] = torch.sqrt(1 - a_) * x0[i] + torch.sqrt(a_) * torch.randn_like(x0[i])
            PEs.append(None)

        if self.level_num > 0:
            g = self.sibling_num
            p_l = 8 // int(g)
            pos = np.arange(p_l, dtype=np.float32) / float(p_l)
            PE = torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
                ).unsqueeze(dim=0).clone().to(x.device).float()
            PE = PE.repeat(1, L // g // p_l, 1)

        # Embed conditions and timesteps
        x0 = self.n_embedder(x0, PEs)               # Previous node embedding
        t = self.t_embedder(t)                   # (N, D)
        a_out = []
        for a_, a_embedder in zip(a, self.a_embedder):
            if not self.real_noa:
                a_out.append(a_embedder(a_))
            else:
                a_out.append(0)
        if self.y_embedder is not None:
            y = self.y_embedder(y, self.training)    # (N, D)
        else:
            y = 0
        c = t + y + sum(a_out)                              # (N, D)

        x = self.input_layer(x)
        ## -1 means using the nearest level GT positions as the positional encoding
        if self.level_num > 0:
            x = x + PE
        if self.level_num == 0 and self.learned_pos_embedding:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.learned_PE(positions)

        for block in self.blocks:
            x = block(x, c, x0, a_out)                      # (N, L, D)
        x = self.output_layer(x, c)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = (np.arange(grid_size, dtype=np.float32) / grid_size) * 2.0 - 1.0
    grid_w = (np.arange(grid_size, dtype=np.float32) / grid_size) * 2.0 - 1.0
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
