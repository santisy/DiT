import math
import torch
from torch import nn
from torch.nn import functional as F
from taming_models import ResnetBlock, Downsample, Upsample, VectorQuantizer2

class reshapeTo3D(nn.Module):
    def __init__(self, grid_size):
        self._g = grid_size
        super(reshapeTo3D, self).__init__()
        
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, C)
        x = x.reshape(B * L, 1, self._g, self._g, self._g)
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
                 *args,
                 **kwargs
                 ):
        super(VAE, self).__init__()
        embed_dim = latent_ch
        self.code_n = 2048
        self.g = grid_size

        # Encoder
        self.input_fc = nn.Sequential(reshapeTo3D(grid_size),
                                      nn.Conv3d(1, in_ch, 3, 1, 1))
        fc1 = []
        # First res
        fc1.extend([ResnetBlock(in_ch) for _ in range(layer_n // 2)])
        fc1.append(Downsample(in_ch, True))
        fc1.append(ResnetBlock(in_ch, in_ch * 2))
        fc1.extend([ResnetBlock(in_ch * 2) for _ in range(layer_n // 2 - 1)])
        self.fc1 = nn.Sequential(*fc1)
        self.quant_conv = nn.Conv3d(in_ch * 2, embed_dim, 1, 1, 0)

        # Decoder
        self.post_quant_conv = nn.Conv3d(embed_dim, in_ch * 2, 1, 1, 0)
        fc4 = []
        fc4.extend([ResnetBlock(in_ch * 2) for _ in range(layer_n // 2)])
        fc4.append(Upsample(in_ch * 2, True))
        fc4.append(ResnetBlock(in_ch * 2, in_ch))
        fc4.extend([ResnetBlock(in_ch, in_ch) for _ in range(layer_n // 2 - 1)])
        self.fc4 = nn.Sequential(*fc4)

        self.output_fc = nn.Sequential(nn.Conv3d(in_ch, 1, 3, 1, 1),
                                       nn.Sigmoid())

        # Quantizer
        self.quantize = VectorQuantizer2(self.code_n, embed_dim, beta=0.25,
                                         remap=None,
                                         sane_index_shape=False,
                                         legacy=False)

    def encode(self, x):
        h = self.input_fc(x)
        h = self.fc1(h)
        h = self.quant_conv(h)
        quant, q_loss, info = self.quantize(h)
        return quant, q_loss, info

    def decode(self, z):
        h = self.post_quant_conv(z)
        h = self.fc4(h)
        return self.output_fc(h)

    def decode_code(self, c):
        B, L, C = c.shape
        quant = self.quantize.get_codebook_entry(c, None)
        quant = quant.reshape(B * L, self.g, self.g, self.g, -1).permute(0, 4, 1, 2, 3).contiguous()
        return self.decode(quant)

    def get_normalized_indices(self, x):
        B, L, C = x.shape
        _, _, info = self.encode(x)
        indices = info[2]
        indices = indices / float(self.code_n)
        indices = indices.reshape(B, L, -1)
        return indices
        
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
    return recon_loss + q_loss.mean()

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
    vae = VAE(4, 64, 16, 5)
    input_data = torch.randn(1, 4, 125)
    output_data, q_loss, info = vae(input_data)
    indices = vae.get_normalized_indices(input_data)
    loss = loss_function(input_data, output_data, q_loss)
    loss.sum().backward() 
    print(loss)
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")

    #encoded = vae.encode_and_reparam(input_data)
    #print(encoded.shape)
