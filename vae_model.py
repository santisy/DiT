import math
import torch
from torch import nn
from torch.nn import functional as F
from taming_models import ResnetBlock, Downsample, Upsample

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
                 latent_ch,
                 grid_size,
                 *args,
                 **kwargs
                 ):
        super(VAE, self).__init__()
        self._g = grid_size

        # Encoder
        self.input_fc = nn.Sequential(reshapeTo3D(grid_size),
                                      nn.Conv3d(1, in_ch, 3, 1, 1))

        self.input_fc_other = nn.Linear(18, in_ch)
        self.fc1_other = nn.Sequential(*[x for x in [nn.GELU(), nn.Linear(in_ch, in_ch), nn.LayerNorm(in_ch)] for _ in range(layer_n)])

        fc1_1 = []
        # First res
        fc1_1.extend([ResnetBlock(in_ch) for _ in range(layer_n // 2)])
        # Second res
        fc1_2 = []
        fc1_2.append(Downsample(in_ch, True))
        fc1_2.append(ResnetBlock(in_ch, in_ch * 2))
        fc1_2.extend([ResnetBlock(in_ch * 2) for _ in range(layer_n // 2 - 1)])

        self.fc1_1 = nn.Sequential(*fc1_1)
        self.fc1_2 = nn.Sequential(*fc1_2)

        self.fc2_mean = nn.Conv3d(in_ch * 2, latent_ch, 1, 1, 0)
        self.fc2_logvar = nn.Conv3d(in_ch * 2, latent_ch, 1, 1, 0)

        # Decoder
        self.fc3 = nn.Conv3d(latent_ch, in_ch * 2, 1, 1, 0)
        fc4_1 = []
        fc4_1.extend([ResnetBlock(in_ch * 2) for _ in range(layer_n // 2)])
        fc4_1.append(Upsample(in_ch * 2, True))
        fc4_1.append(ResnetBlock(in_ch * 2, in_ch))

        fc4_2 = []
        fc4_2.extend([ResnetBlock(in_ch, in_ch) for _ in range(layer_n // 2 - 1)])

        self.fc4_1 = nn.Sequential(*fc4_1)
        self.fc4_2 = nn.Sequential(*fc4_2)

        self.fc4_other = nn.Sequential(*[x for x in [nn.GELU(), nn.Linear(in_ch, in_ch), nn.LayerNorm(in_ch)] for _ in range(layer_n)])
        self.output_fc_other = nn.Sequential(nn.Linear(in_ch, 18), nn.Sigmoid())

        self.output_fc = nn.Sequential(nn.Conv3d(in_ch, 1, 3, 1, 1),
                                       nn.Sigmoid())

    def encode(self, x):
        x_other = x[:, :, self._g ** 3:]
        B, L, C = x_other.shape
        h_other = self.input_fc_other(x_other.reshape(B * L, C))
        h_other = self.fc1_other(h_other).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        h = self.input_fc(x[:, :, :self._g ** 3])
        h = self.fc1_1(h) + h_other
        h = self.fc1_2(h)

        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        h = self.fc3(z)
        h = self.fc4_1(h)
        h_other = F.adaptive_avg_pool3d(h, (1, 1, 1))
        B = h_other.shape[0]
        h_other = h_other.reshape(B, -1)
        h_other = self.fc4_other(h_other)
        #h = h - h_other.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out_other = self.output_fc_other(h_other)
        h = self.fc4_2(h)
        out = self.output_fc(h)
        return out, out_other

    def encode_and_reparam(self, x):
        B, L, C = x.shape
        mean, logvar = self.encode(x)
        out = self.reparameterize(mean, logvar)
        out = out.reshape(B, L, -1)

        return out
        
    def forward(self, x):
        B, L, C = x.shape

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out, out_other =  self.decode(z)

        out = out.reshape(B, L, C - 18)
        out_other = out_other.reshape(B, L, 18)
        mean = mean.reshape(B, L, -1)
        logvar = logvar.reshape(B, L, -1)

        return torch.cat([out, out_other], dim=-1), mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar, kl_weight=1e-6):
    b = x.size(0) * x.size(1)
    # Recon loss at regular grid
    recon_loss = F.l1_loss(recon_x[:, :, :125],
                           x[:, :, :125], reduction="sum") / b
    # Recon other
    recon_loss_other = F.l1_loss(recon_x[:, :, 125:],
                                 x[:, :, 125:], reduction="sum") / b * 10.0
    # Regularize angular encoding
    ra_loss = ((recon_x[:, :, 125] ** 2.0 + recon_x[:, :, 126] ** 2.0 - 1.0).abs().sum() / b + 
               (recon_x[:, :, 127] ** 2.0 + recon_x[:, :, 128] ** 2.0 - 1.0).abs().sum() / b +
               (recon_x[:, :, 129] ** 2.0 + recon_x[:, :, 130] ** 2.0 - 1.0).abs().sum() / b +
               (recon_x[:, :, 131] ** 2.0 + recon_x[:, :, 132] ** 2.0 - 1.0).abs().sum() / b
               ) * 10
    # KL divergence
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b
    return recon_loss + recon_loss_other + ra_loss + KLD * kl_weight

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
    vae = VAE(4, 64, 2, 5)
    input_data = torch.randn(1, 4, 143)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    loss.sum().backward() 
    print(loss)
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")

    encoded = vae.encode_and_reparam(input_data)
    print(encoded.shape)
