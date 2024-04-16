import math
import torch
from torch import nn
from torch.nn import functional as F

class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModule, self).__init__()
        #self.embed = nn.Linear(token_size, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                       dim_feedforward=d_model * 4,
                                                       nhead=nhead,
                                                       dropout=dropout,
                                                       activation="gelu",
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
    
    def forward(self, x):
        #x = self.embed(x) * math.sqrt(self.d_model)  # Embedding and scaling
        x = self.transformer(x)  # Transformer encoder processes batches first
        return x

class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_layers=8,
                 token_n=16,
                 nhead=4
                 ):
        super(VAE, self).__init__()
        self.token_n = token_n
        self.d_model = d_model = input_dim * 16 // token_n
        self.embed_dim = d_model * token_n
        self.latent_dim = latent_dim

        self.input_layer = nn.Linear(input_dim, self.embed_dim)
        self.encoder = TransformerModule(d_model, nhead, num_layers)
        self.decoder = TransformerModule(d_model, nhead, num_layers)  # Assuming symmetry in the architecture

        self.fc_mu = nn.Linear(d_model * token_n, latent_dim)
        self.fc_var = nn.Linear(d_model * token_n, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, d_model * token_n)

        self.output_layer = nn.Linear(self.embed_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decode
        B = z.size(0)
        z = self.fc_decode(z)
        z = z.reshape(B, -1, self.d_model)
        z = self.decoder(z)

        # Output
        z = z.reshape(B, -1)
        out = self.output_layer(z)
        return out

    def encode_and_reparam(self, x):
        B = x.size(0)
        # Reshape and embed
        x = self.input_layer(x)
        x = x.reshape(B, -1, self.d_model)

        # Encode
        encoded = self.encoder(x)

        # Reparameterize
        encoded = encoded.reshape(B, -1)  # Pooling over the sequence
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        z = z.reshape(B, self.latent_dim)

        return z

    def forward(self, x):
        B = x.size(0)
        # Reshape and embed
        x = self.input_layer(x)
        x = x.reshape(B, -1, self.d_model)

        # Encode
        encoded = self.encoder(x)

        # Reparameterize
        encoded = encoded.reshape(B, -1)  # Pooling over the sequence
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)

        # Decode
        z = self.fc_decode(z)
        z = z.reshape(B, -1, self.d_model)
        z = self.decoder(z)

        # Output
        z = z.reshape(B, -1)
        out = self.output_layer(z)

        return out, mu, log_var

#class VAE(nn.Module):
#    def __init__(self,
#                 layer_n,
#                 input_dim,
#                 hidden_dim,
#                 latent_dim,
#                 ):
#        super(VAE, self).__init__()
#        # Encoder
#        self.input_fc = nn.Linear(input_dim, hidden_dim)
#        self.fc1 = nn.Sequential(*[x for x in [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)] for _ in range(layer_n)])
#        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
#        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
#        # Decoder
#        self.fc3 = nn.Linear(latent_dim, hidden_dim)
#        self.fc4 = nn.Sequential(*[x for x in [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)] for _ in range(layer_n)])
#        self.output_fc = nn.Linear(hidden_dim, input_dim)
#
#    def encode(self, x):
#        x = self.input_fc(x)
#        h = self.fc1(x)
#        return self.fc2_mean(h), self.fc2_logvar(h)
#
#    def reparameterize(self, mean, logvar):
#        std = torch.exp(0.5 * logvar)
#        eps = torch.randn_like(std)
#        return mean + eps*std
#
#    def decode(self, z):
#        h = self.fc3(z)
#        h = self.fc4(h)
#        return self.output_fc(h)
#
#    def encode_and_reparam(self, x):
#        B, L, C = x.shape
#        x = x.reshape(B * L, C)
#
#        mean, logvar = self.encode(x)
#        out = self.reparameterize(mean, logvar)
#        out = out.reshape(B, L, -1)
#
#        return out
#        
#    def forward(self, x):
#        B, L, C = x.shape
#        x = x.reshape(B * L, C)
#
#        mean, logvar = self.encode(x)
#        z = self.reparameterize(mean, logvar)
#        out =  self.decode(z)
#
#        out = out.reshape(B, L, C)
#        mean = mean.reshape(B, L, -1)
#        logvar = logvar.reshape(B, L, -1)
#
#        return out, mean, logvar

# Loss function
def loss_function(recon_x, x, mean, logvar, kl_weight=1e-6):
    b = x.size(0)
    recon = F.l1_loss(recon_x, x, reduction="sum") / b
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
    vae = VAE(128, 32)
    #print(vae)
    input_data = torch.randn(16, 128)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar)
    print(loss)

    latent = vae.encode_and_reparam(input_data)
    print(f"latent shape is {latent.shape}")
    out = vae.decode(latent)
    print(f"Output data is {out.shape}")
