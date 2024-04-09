import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, layer_n, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Sequential(*[x for x in [nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n)])
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Sequential(*[x for x in [nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n)])
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = self.input_fc(x)
        h1 = self.fc1(x)
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        h = self.fc3(z)
        h = self.fc4(h)
        return self.output_fc(h)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar, kl_weight=1e-6):
    b = x.size(0) * x.size(1)
    recon = F.mse_loss(recon_x, x, reduction="sum") / b
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b
    return recon + KLD * kl_weight

if __name__ == "__main__":
    vae = VAE(8, 128, 1024, 64)
    input_data = torch.randn(4, 32, 128)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    print(loss)
