import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_module import PreNormSelfAttention
from transformer_module import GEGLU


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

class MiniCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout,
                                                batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, cond):
        # Perform cross-attention
        src2 = self.cross_attn(src, cond, cond)[0]
        # Apply dropout and add residual connection
        src = src + self.dropout(src2)
        return src

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
                 condition_node_dim=[],
                 flow_flag=False,
                 no_a_embed=False,
                 rescale_flag=False,
                 real_noa=False,
                 selftt=False,
                 learn_sigma=False,
                 out_ch=None,
                 reg_flag=False,
                 num_classes=None,
                 uncond_flag=False,
                 cross_attn=False,
                 **kwargs
                 ):

        super(PlainModel, self).__init__()
        self.in_ch = in_channels * sibling_num
        self.embed_dim = hidden_size
        self.sibling_num = sibling_num
        self.condition_node_dim = condition_node_dim
        self.no_a_embed = no_a_embed
        self.flow_flag = flow_flag
        self.rescale_flag = rescale_flag
        self.real_noa = real_noa
        self.reg_flag = reg_flag
        self.uncond_flag = uncond_flag
        self.cross_attn = cross_attn

        # Class conditional related
        if num_classes is not None:
            y_embed_dim = 256
            self.class_cond_flag = True
            self.class_embedding = nn.Embedding(num_classes, y_embed_dim)
            nn.init.normal_(self.class_embedding.weight, std=0.02)
            self.y_embed = nn.Sequential(
                nn.Linear(y_embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            ) 
        else:
            self.class_cond_flag = False


        if not selftt:
            layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                               nhead=num_heads,
                                               norm_first=True,
                                               dim_feedforward=int(mlp_ratio * hidden_size),
                                               dropout=0.1,
                                               batch_first=True)
            self.net = nn.TransformerEncoder(layer, depth, nn.LayerNorm(self.embed_dim))
            if cross_attn:
                self.cross_attn_layers = nn.ModuleList()
                for _ in range(depth // 4):
                    self.cross_attn_layers.append(MiniCrossAttention(self.embed_dim,
                                                                     nhead=num_heads,
                                                                     batch_first=True,
                                                                     dropout=0.1))
        else:
            self.net = nn.Sequential(*[PreNormSelfAttention(self.embed_dim,
                                                            num_heads,
                                                            self.embed_dim // num_heads,
                                                            mult=int(mlp_ratio),
                                                            dropout=0.1) for _ in range(depth)])

        self.p_embed = nn.Sequential(
            nn.Linear(self.in_ch, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ) 

        if not reg_flag:
            self.time_embed = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        out_ch = self.in_ch if out_ch is None else out_ch
        out_ch = out_ch if not learn_sigma else out_ch * 2
        if reg_flag:
            out_ch = out_ch * sibling_num

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, int(out_ch)),
        )

        if len(condition_node_dim) > 0 and not reg_flag and not uncond_flag:
            self.a_embed_list = nn.ModuleList()
            self.c_embed_list = nn.ModuleList()
            for c_nd in condition_node_dim:
                if not real_noa:
                    self.a_embed_list.append(nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim),
                        nn.LayerNorm(self.embed_dim),
                        nn.SiLU(),
                        nn.Linear(self.embed_dim, self.embed_dim))
                    )
                else:
                    self.a_embed_list.append(nn.Identity())
                self.c_embed_list.append(nn.Sequential(
                    nn.Linear(c_nd, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.embed_dim, self.embed_dim))
                )

        return

       
    def forward(self, x, timesteps, y=None, a=[], x0=[], **kwargs):

        B, L, C = x.shape
        if self.sibling_num > 1:
            x = x.reshape(B, L // self.sibling_num, -1)
            L_x = L // self.sibling_num
        else:
            L_x = L

        other_embed_accumulate = 0
        if len(self.condition_node_dim) > 0 and not self.reg_flag and not self.uncond_flag:
            # Noise augmentation level `a`, and previous condition embedding `c`
            for a_, xc_, a_embed, c_embed in zip(
                a, x0, self.a_embed_list, self.c_embed_list):
                if not self.real_noa:
                    a_embeded = a_embed(sincos_embedding(a_, self.embed_dim)).unsqueeze(1)
                    other_embed_accumulate = other_embed_accumulate + a_embeded
                xc_embeded = c_embed(xc_)
                L_xc = xc_embeded.size(1)
                xc_embeded = torch.repeat_interleave(xc_embeded, L_x // L_xc, dim=1)
                other_embed_accumulate = other_embed_accumulate + xc_embeded

            p_l = 8 // self.sibling_num
            pos = torch.arange(p_l).unsqueeze(dim=0).to(x.device)
            PE = sincos_embedding(pos, self.embed_dim)
            PE = PE.repeat(1, L // self.sibling_num // p_l, 1)
        else:
            PE = 0

        # y (class lebel embed)
        if self.class_cond_flag:
            y_embeds = self.class_embedding(y)
            y_embeds = self.y_embed(y_embeds)
            y_embeds = y_embeds.unsqueeze(dim=1)
        else:
            y_embeds = 0

        # t (timestep) embed
        if self.flow_flag:
            timesteps = (timesteps * 1000).floor().to(torch.int64)
        if not self.reg_flag:
            time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)  
        else:
            time_embeds = 0

        # x (input) embed
        x_embeds = self.p_embed(x)
        tokens = x_embeds + other_embed_accumulate + PE

        """ forward pass """
        if not self.cross_attn:
            tokens = tokens + time_embeds + y_embeds
            output = self.net(tokens)
        else:
            x_ = tokens
            context = time_embeds + y_embeds
            for i, layer in enumerate(self.net.layers):
                if i % 4 == 0:
                    x_ = self.cross_attn_layers[i // 4](x_, context)
                x_ = layer(x_)
            output = self.net.norm(x_)

        pred = self.fc_out(output)
        pred = pred.reshape(B, L // self.sibling_num, self.sibling_num, -1)
        pred = pred.reshape(B, L, -1)
        return pred


if __name__ == "__main__":
    from torch.cuda.amp import autocast
    net = PlainModel(4,
                    depth=12,
                    num_heads=16,
                    hidden_size=512,
                    no_a_embed=True,
                    real_noa=True,
                    num_classes=1,
                    cross_attn=True,
                    selftt=False).cuda()

    t = torch.randint(0, 1024, (4,)).cuda()
    x = torch.randn(4, 256, 4).cuda()
    y = torch.tensor([0,] * 4).long().cuda()

    with autocast(enabled=True):
        out = net(x, t, y=y)
    out.sum().backward()
    print(out.shape)
    
    # Check for parameters that did not collect gradients
    for name, param in net.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter '{name}' did not collect a gradient.")
