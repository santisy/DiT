import math
import torch
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # This makes it [1, max_len, d_model]

    def forward(self, x):
        """
        Adds positional encoding to the input batch considering batch first.
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.encoding[:, :x.size(1)].to(x.device)

class VAE(nn.Module):
    def __init__(self, layer_n, input_dim, hidden_dim, latent_dim, nhead, num_tokens):
        super(VAE, self).__init__()
        self.token_size = token_size = hidden_dim // num_tokens
        self.num_tokens = num_tokens

        input_layers = [nn.Linear(input_dim, hidden_dim)]
        input_layers.extend([x for x in [nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n // 2)])
        self.input_fc = nn.Sequential(*input_layers)
        self.pos_encoder = PositionalEncoding(token_size)

        encoder_layer = nn.TransformerEncoderLayer(token_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, layer_n // 2)

        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(token_size, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, layer_n // 2)
        output_layers = [x for x in [nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)] for _ in range(layer_n // 2)]
        output_layers.extend([nn.Linear(hidden_dim, input_dim), nn.Sigmoid()])
        self.output_fc = nn.Sequential(*output_layers)

    def encode(self, x):
        x = self.input_fc(x)
        x = x.view(-1, self.num_tokens, self.token_size)  # Reshape to tokens
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten back before linear layers
        return self.fc2_mean(x), self.fc2_logvar(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.num_tokens, self.token_size)  # Reshape for decoding
        z = self.transformer_decoder(z, z)
        z = z.view(z.size(0), -1)  # Flatten for output
        return self.output_fc(z)

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B * L, C)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        out = out.reshape([B, L, C])
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
    hidden_dim = 361 * 16 // (4 * 8) * 4 * 8
    vae = VAE(4, 361, hidden_dim, 128, 4, 8)
    input_data = torch.randn(1, 4, 361)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    loss.sum().backward() 
    print(loss)
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")
