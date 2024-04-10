import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class MLPSkip(nn.Module):
    def __init__(self, hidden_dim):
        super(MLPSkip, self).__init__()
        self.layer = nn.Sequential(nn.SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim))
    def forward(self, x):
        return self.layer(x)

class Reshape(nn.Module):
    def __init__(self, ch):
        self.ch = ch
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.reshape(-1, self.ch)

class VAE(nn.Module):
    def __init__(self, layer_n, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        self.fc1 = nn.Sequential(*[MLPSkip(hidden_dim) for _ in range(layer_n)])
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Sequential(*[MLPSkip(hidden_dim) for _ in range(layer_n)])

        self.output_fc = nn.Linear(hidden_dim, input_dim)

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
        mean, logvar = self.encode(x)
        out = self.reparameterize(mean, logvar)
        return out
        
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar, kl_weight=1e-6):
    b = x.size(0) * x.size(1)
    recon = F.mse_loss(recon_x, x, reduction="sum") / b
    # KL divergence
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b
    return recon + KLD * kl_weight

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
    vae = VAE(8, 128, 1024, 64)
    input_data = torch.randn(4, 32, 128)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    print(loss)
