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
        self.fc2_mean = nn.Conv3d(in_ch * 2, latent_ch, 1, 1, 0)
        self.fc2_logvar = nn.Conv3d(in_ch * 2, latent_ch, 1, 1, 0)

        # Decoder
        self.fc3 = nn.Conv3d(latent_ch, in_ch * 2, 1, 1, 0)
        fc4 = []
        fc4.extend([ResnetBlock(in_ch * 2) for _ in range(layer_n // 2)])
        fc4.append(Upsample(in_ch * 2, True))
        fc4.append(ResnetBlock(in_ch * 2, in_ch))
        fc4.extend([ResnetBlock(in_ch, in_ch) for _ in range(layer_n // 2 - 1)])
        self.fc4 = nn.Sequential(*fc4)

        self.output_fc = nn.Sequential(nn.Conv3d(in_ch, 1, 3, 1, 1),
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
        mean, logvar = self.encode(x)
        out = self.reparameterize(mean, logvar)
        out = out.reshape(B, L, -1)

        return out
        
    def forward(self, x):
        B, L, C = x.shape

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out =  self.decode(z)

        out = out.reshape(B, L, C)
        mean = mean.reshape(B, L, -1)
        logvar = logvar.reshape(B, L, -1)

        return out, mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar, kl_weight=1e-6):
    b = x.size(0) * x.size(1)
    recon_loss = F.l1_loss(recon_x, x, reduction="sum") / b
    # KL divergence
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b
    return recon_loss + KLD * kl_weight

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
    vae = VAE(4, 64, 2, 7)
    input_data = torch.randn(1, 4, 343)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    loss.sum().backward() 
    print(loss)
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")

    encoded = vae.encode_and_reparam(input_data)
    print(encoded.shape)
