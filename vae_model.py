import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from taming_models import ResnetBlock, Downsample, Upsample, VectorQuantizer2
from taming_models import Downsample2, Upsample2, Downsample3, Upsample3
from vector_quantize_pytorch import VectorQuantize

from functools import partial

class reshapeTo3D(nn.Module):
    def __init__(self, grid_size):
        self._g = grid_size
        super(reshapeTo3D, self).__init__()
        
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, C)
        x = x.reshape(B * L, 1, self._g, self._g, self._g)
        return x

class reshapeTo1D(nn.Module):
    def __init__(self, grid_size):
        self._g = grid_size
        super(reshapeTo1D, self).__init__()
        
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, 1, C).contiguous()
        return x

class VAELinear(nn.Module):
    def __init__(self,
                 layer_n,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 *args,
                 **kwargs
                 ):
        super(VAELinear, self).__init__()
        # Encoder
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Sequential(*[x for x in [nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n)])
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Sequential(*[x for x in [nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n)])
        self.output_fc = nn.Sequential(nn.Linear(hidden_dim, input_dim),
                                       nn.Sigmoid())

    def encode(self, x):
        x = self.input_fc(x)
        h = self.fc1(x)
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        h = self.fc3(z)
        h = self.fc4(h)
        return self.output_fc(h)

    def encode_and_reparam(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, C)

        mean, logvar = self.encode(x)
        out = self.reparameterize(mean, logvar)
        out = out.reshape(B, L, -1)

        return out
        
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, C)

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out =  self.decode(z)

        out = out.reshape(B, L, C)
        mean = mean.reshape(B, L, -1)
        logvar = logvar.reshape(B, L, -1)

        return out, mean, logvar

class VAE(nn.Module):
    def __init__(self,
                 layer_n,
                 in_ch,
                 latent_ch=16,
                 grid_size=5,
                 quant_code_n=2048,
                 quant_version="v0",
                 quant_heads=1,
                 downsample_n=2,
                 level_num=2,
                 kl_flag=False,
                 *args,
                 **kwargs
                 ):
        super(VAE, self).__init__()
        if level_num == 2:
            downsample_n = min(2, downsample_n)
        self.embed_dim = embed_dim = latent_ch
        self.code_n = quant_code_n
        self.g = grid_size
        self.m_ = math.ceil(grid_size / (2 ** downsample_n))
        self.quant_version = quant_version
        self.quant_heads = quant_heads
        self.downsample_n = downsample_n
        self.level_num = level_num
        self.kl_flag = kl_flag

        upsample_list = []
        downsample_list = []
        if level_num == 2:
            conv = nn.Conv3d
            reshapeModule = reshapeTo3D
            downsample_list.append(partial(Downsample, with_conv=True))
            downsample_list.append(partial(Downsample2, with_conv=True))
            upsample_list.append(partial(Upsample2, with_conv=True))
            upsample_list.append(partial(Upsample, with_conv=True))
        else:
            conv = nn.Conv1d
            reshapeModule = reshapeTo1D
            downsample_list.extend([partial(Downsample3, with_conv=True) for _ in range(downsample_n)])
            upsample_list.extend([partial(Upsample3, with_conv=True) for _ in range(downsample_n)])

        resnet_block = partial(ResnetBlock, conv=conv)

        # Encoder
        self.input_fc = nn.Sequential(reshapeModule(grid_size),
                                      conv(1, in_ch, 3, 1, 1))
        fc1 = []
        # First res
        ch = None
        for i in range(downsample_n):
            ch = int(in_ch * 2 ** i)
            fc1.extend([resnet_block(ch) for _ in range(layer_n)])
            fc1.append(downsample_list[i](ch))
            fc1.append(resnet_block(ch, ch * 2))
        ch = ch * 2
        ch_last = ch
        fc1.extend([resnet_block(ch) for _ in range(layer_n)])
        self.fc1 = nn.Sequential(*fc1)

        # Decoder
        fc4 = []
        for i in range(downsample_n):
            fc4.extend([resnet_block(ch) for _ in range(layer_n)])
            fc4.append(upsample_list[i](ch))
            fc4.append(resnet_block(ch, ch // 2))
            ch = ch // 2

        fc4.extend([resnet_block(ch) for _ in range(layer_n)])
        self.fc4 = nn.Sequential(*fc4)

        self.output_fc = nn.Sequential(conv(in_ch, 1, 3, 1, 1),
                                       nn.Sigmoid())

        # Quantizer
        if not kl_flag:
            print(f"\033[92m Use quant version {quant_version}.\033[00m")
            if quant_version == "v0":
                self.quantize = nn.ModuleList()
                for _ in range(quant_heads):
                    self.quantize.append(VectorQuantizer2(self.code_n, embed_dim, beta=0.25,
                                                        remap=None,
                                                        sane_index_shape=False,
                                                        legacy=False))
                self.quant_conv = conv(ch_last // quant_heads, embed_dim, 1, 1, 0)
                self.post_quant_conv = conv(embed_dim * quant_heads, ch_last, 1, 1, 0)
            elif quant_version == "v1":
                self.quantize = VectorQuantize(dim = ch_last,
                                               codebook_size = self.code_n,
                                               codebook_dim=embed_dim,
                                               use_cosine_sim=True,
                                               separate_codebook_per_head = True,
                                               heads=quant_heads)
        else:
            self.fc2_mean = conv(ch_last, embed_dim, 1, 1, 0)
            self.fc2_logvar = conv(ch_last, embed_dim, 1, 1, 0)
            self.post_quant_conv = conv(embed_dim, ch_last, 1, 1, 0)


    def get_code_book_n(self):
        return self.code_n

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def encode(self, x):
        B, L, _ = x.shape

        h = self.input_fc(x)
        h = self.fc1(h)
        if self.kl_flag:
            mean = self.fc2_mean(h)
            logvar = self.fc2_logvar(h)
            out = self.reparameterize(mean, logvar)
            q_loss = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / (B * L) * 1e-6
            return out, q_loss, None

        if self.quant_version == "v0":
            indices_list = []
            quant_list = []
            q_loss = 0
            for q, h_ in zip(self.quantize, torch.chunk(h, self.quant_heads, dim=1)):
                h_ = self.quant_conv(h_)
                quant_, q_loss_, info = q(h_)
                q_loss += q_loss_
                indices_list.append(info[2])
                quant_list.append(quant_)
            indices = torch.stack(indices_list, dim=-1)
            quant = torch.cat(quant_list, dim=1)
        elif self.quant_version == "v1":
            if self.level_num == 2:
                z = rearrange(h, 'b c h w d -> b (h w d) c').contiguous()
            else:
                z = rearrange(h, 'b c d -> b d c').contiguous()
            quant, indices, q_loss = self.quantize(z)
            quant = quant.permute(0, 2, 1).view(h.shape).contiguous()
        return quant, q_loss, indices

    def decode(self, z):
        if self.quant_version == "v0" or self.kl_flag:
            h = self.post_quant_conv(z)
        else:
            h = z
        h = self.fc4(h)
        return self.output_fc(h)

    def decode_code(self, c: torch.Tensor):
        B, L, C = c.shape
        c = (c.clamp_(0, 1) * self.code_n).floor().long()
        c = torch.clamp(c, 0, self.code_n - 1)
        c = c.reshape(B * L, C // 2, 2)
        if self.quant_version == "v0":
            quant_list = []
            for q, c_ in zip(self.quantize, torch.chunk(c, self.quant_heads, dim=-1)):
                quant_ = q.get_codebook_entry(c_, None)
                if self.level_num == 2:
                    quant_ = quant_.reshape(B * L, self.m_, self.m_, self.m_, -1).permute(0, 4, 1, 2, 3).contiguous()
                else:
                    quant_ = quant_.reshape(B * L, -1, self.embed_dim).permute(0, 2, 1).contiguous()
                quant_list.append(quant_)
            quant = torch.cat(quant_list, dim=1)
        elif self.quant_version == "v1":
            quant = self.quantize.get_output_from_indices(c)
        out = self.decode(quant)
        out = out.reshape(B, L, -1)
        return out

    def get_normalized_indices(self, x):
        B, L, C = x.shape
        if not self.kl_flag:
            _, _, indices = self.encode(x)
            indices = indices / float(self.code_n)
            indices = indices.reshape(B, L, -1)
            return indices
        else:
            out, _, _ = self.encode(x)
            out = out.reshape(B, L, -1)
            return out

        
    def forward(self, x):
        B, L, C = x.shape

        quant, q_loss, info = self.encode(x)
        out =  self.decode(quant)

        out = out.reshape(B, L, C)

        return out, q_loss, info

# Loss function
def loss_function(recon_x, x, q_loss):
    b = x.size(0) * x.size(1)
    recon_loss = F.l1_loss(recon_x, x, reduction="sum") / b
    # KL divergence
    return recon_loss + q_loss.mean(), recon_loss

class OnlineVariance(object):
    def __init__(self, num_features):
        self.n = 0
        self.mean = torch.zeros(num_features)
        self.M2 = torch.zeros(num_features)
        self.num_features = num_features

    def update(self, batch):
        # Expect batch to be of shape [N, C]
        batch_mean = torch.mean(batch, dim=0)
        batch_count = batch.size(0)
        
        if self.n == 0:
            self.mean = batch_mean
        else:
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / (self.n + batch_count)
            self.M2 += torch.sum((batch - batch_mean.unsqueeze(0))**2 + (delta**2) * self.n * batch_count / (self.n + batch_count), dim=0)
        
        self.n += batch_count

    @property
    def std(self):
        if self.n < 2:
            return float('nan') * torch.ones(self.num_features)  # Return nan if less than 2 samples
        std = torch.sqrt(self.M2 / (self.n - 1)).detach().cpu().numpy()
        return std

if __name__ == "__main__":
    vae = VAE(2, 64, 1, 10,
              downsample_n=3,
              quant_code_n=512,
              quant_heads=1,
              quant_version="v0",
              kl_flag=True,
              level_num=1)
    input_data = torch.randn(1, 4, 128)
    output_data, q_loss, _= vae(input_data)
    loss, _ = loss_function(input_data, output_data, q_loss)
    loss.sum().backward() 
    print(loss)
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")

    codes = vae.get_normalized_indices(input_data)
    print(codes.shape)
