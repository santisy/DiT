import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class PlainModel(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """

    def __init__(self,
                 in_channels,
                 depth=12,
                 num_heads=16,
                 hidden_size=1024,
                 mlp_ratio=2,
                 sibling_num=1,
                 **kwargs
                 ):

        super(PlainModel, self).__init__()
        self.in_ch = in_channels * sibling_num
        self.embed_dim = hidden_size
        self.sibling_num = sibling_num

        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                           nhead=num_heads,
                                           norm_first=True,
                                           dim_feedforward=int(mlp_ratio * hidden_size),
                                           dropout=0.1,
                                           batch_first=True)
        self.net = nn.TransformerEncoder(layer, depth, nn.LayerNorm(self.embed_dim))

        self.p_embed = nn.Sequential(
            nn.Linear(self.in_ch, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ) 

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.in_ch),
        )

        return

       
    def forward(self, x, timesteps, y=None, **kwargs):
        B, L, C = x.shape

        if self.sibling_num > 1:
            x = x.reshape(B, L // self.sibling_num, -1)

        """ forward pass """
        bsz = timesteps.size(0)
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)  
        x_embeds = self.p_embed(x)
    
        tokens = x_embeds + time_embeds
        output = self.net(src=tokens)
        pred = self.fc_out(output)
        pred = pred.reshape(B, L, C)

        return pred
