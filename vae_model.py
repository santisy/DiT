import math
import torch
from torch import nn
from torch.nn import functional as F

class VAEVanilla(nn.Module):
    def __init__(self,
                 layer_n,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 *args,
                 **kwargs
                 ):
        super(VAEVanilla, self).__init__()
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
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 level_num=0
                 ):

        super(VAE, self).__init__()
        # Encoder
        self.level_num = level_num
        self.grid_m = 7 if level_num == 0 else 5
        
        hidden_count = 0

        unit = lambda x: [nn.GELU(), nn.Dropout(0.1), nn.Linear(x, x), nn.LayerNorm(x)]

        # Grid values encode branch
        self.hidden_v = hidden_v = self.grid_m ** 3 * 16 if level_num != 0 else self.grid_m ** 3 * 8
        fc_grid_v_encode = [nn.Linear(self.grid_m ** 3, hidden_v)]
        fc_grid_v_encode = fc_grid_v_encode + [x for x in unit(hidden_v) for _ in range(layer_n // 2)] 
        self.fc_grid_v_encode = nn.Sequential(*fc_grid_v_encode)
        fc_grid_v_decode = [x for x in unit(hidden_v) for _ in range(layer_n // 2)] 
        fc_grid_v_decode = fc_grid_v_decode + [nn.Linear(hidden_v, self.grid_m ** 3), nn.Sigmoid()]
        self.fc_grid_v_decode = nn.Sequential(*fc_grid_v_decode)
        hidden_count += hidden_v

        # Grid orientation (angular encoded)
        self.hidden_a = hidden_a = 256
        fc_grid_a_encode = [nn.Linear(8, hidden_a)]
        fc_grid_a_encode = fc_grid_a_encode + [x for x in unit(hidden_a) for _ in range(layer_n // 2)] 
        self.fc_grid_a_encode = nn.Sequential(*fc_grid_a_encode)
        fc_grid_a_decode = [x for x in unit(hidden_a) for _ in range(layer_n // 2)] 
        fc_grid_a_decode = fc_grid_a_decode + [nn.Linear(hidden_a, 8), nn.Sigmoid()]
        self.fc_grid_a_decode = nn.Sequential(*fc_grid_a_decode)
        hidden_count += hidden_a

        # Grid scale encode branch
        self.hidden_s = hidden_s = 256
        fc_grid_s_encode = [nn.Linear(3, hidden_s)]
        fc_grid_s_encode = fc_grid_s_encode + [x for x in unit(hidden_s) for _ in range(layer_n // 2)] 
        self.fc_grid_s_encode = nn.Sequential(*fc_grid_s_encode)
        fc_grid_s_decode = [x for x in unit(hidden_s) for _ in range(layer_n // 2)] 
        fc_grid_s_decode = fc_grid_s_decode + [nn.Linear(hidden_s, 3), nn.Sigmoid()]
        self.fc_grid_s_decode = nn.Sequential(*fc_grid_s_decode)
        hidden_count += hidden_s

        # Grid positions
        self.hidden_p = hidden_p = 256
        fc_grid_p_encode = [nn.Linear(3, hidden_p)]
        fc_grid_p_encode = fc_grid_p_encode + [x for x in unit(hidden_p) for _ in range(layer_n // 2)] 
        self.fc_grid_p_encode = nn.Sequential(*fc_grid_p_encode)
        fc_grid_p_decode = [x for x in unit(hidden_p) for _ in range(layer_n // 2)] 
        fc_grid_p_decode = fc_grid_p_decode + [nn.Linear(hidden_p, 3), nn.Sigmoid()]
        self.fc_grid_p_decode = nn.Sequential(*fc_grid_p_decode)
        hidden_count += hidden_p

        if level_num == 0:
            self.hidden_vp = hidden_vp = 256
            fc_voxel_p_encode = [nn.Linear(3, hidden_vp)]
            fc_voxel_p_encode = fc_voxel_p_encode + [x for x in unit(hidden_vp) for _ in range(layer_n // 2)] 
            self.fc_voxel_p_encode = nn.Sequential(*fc_voxel_p_encode)
            fc_voxel_p_decode = [x for x in unit(hidden_vp) for _ in range(layer_n // 2)] 
            fc_voxel_p_decode = fc_voxel_p_decode + [nn.Linear(hidden_vp, 3), nn.Sigmoid()]
            self.fc_voxel_p_decode = nn.Sequential(*fc_voxel_p_decode)
            hidden_count += hidden_vp

            self.hidden_vl = hidden_vl = 128
            fc_voxel_l_encode = [nn.Linear(1, hidden_vl)]
            fc_voxel_l_encode = fc_voxel_l_encode + [x for x in unit(hidden_vl) for _ in range(layer_n // 2)] 
            self.fc_voxel_l_encode = nn.Sequential(*fc_voxel_l_encode)
            fc_voxel_l_decode = [x for x in unit(hidden_vl) for _ in range(layer_n // 2)] 
            fc_voxel_l_decode = fc_voxel_l_decode + [nn.Linear(hidden_vl, 1), nn.Sigmoid()]
            self.fc_voxel_l_decode = nn.Sequential(*fc_voxel_l_decode)
            hidden_count += hidden_vl

        self.combined_fc = nn.Linear(hidden_count, hidden_dim)
        self.fc_combined_encode = nn.Sequential(*[x for x in unit(hidden_dim) for i in range(layer_n // 2)])
        self.fc_combined_decode = nn.Sequential(*[x for x in unit(hidden_dim) for i in range(layer_n // 2)])
        self.split_fc = nn.Linear(hidden_dim, hidden_count)

        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.map_back = nn.Linear(latent_dim, hidden_dim)

    def encode(self, x):
        encode_collect = []
        if self.level_num == 0: 
            encode_collect.append(self.fc_grid_v_encode(x[:, :self.grid_m ** 3]))
            encode_collect.append(self.fc_grid_a_encode(x[:, self.grid_m ** 3: self.grid_m ** 3 + 8]))
            encode_collect.append(self.fc_grid_s_encode(x[:, self.grid_m ** 3 + 8: self.grid_m ** 3 + 11]))
            encode_collect.append(self.fc_voxel_l_encode(x[:, self.grid_m ** 3 + 11: self.grid_m ** 3 + 12]))
            encode_collect.append(self.fc_grid_p_encode(x[:, self.grid_m ** 3 + 12: self.grid_m ** 3 + 15]))
            encode_collect.append(self.fc_voxel_p_encode(x[:, self.grid_m ** 3 + 15: self.grid_m ** 3 + 18]))
        else:
            encode_collect.append(self.fc_grid_v_encode(x[:, :self.grid_m ** 3]))
            encode_collect.append(self.fc_grid_a_encode(x[:, self.grid_m ** 3: self.grid_m ** 3 + 8]))
            encode_collect.append(self.fc_grid_s_encode(x[:, self.grid_m ** 3 + 8: self.grid_m ** 3 + 11]))
            encode_collect.append(self.fc_grid_p_encode(x[:, self.grid_m ** 3 + 11: self.grid_m ** 3 + 14]))

        h = self.combined_fc(torch.cat(encode_collect, dim=1))
        h = self.fc_combined_encode(h)

        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        h = self.map_back(z)
        h = self.fc_combined_decode(h)
        h = self.split_fc(h)
        out_collect = []
        if self.level_num == 0: 
            hc = 0
            out_collect.append(self.fc_grid_v_decode(h[:, hc: hc + self.hidden_v]))
            hc += self.hidden_v
            out_collect.append(self.fc_grid_a_decode(h[:, hc: hc + self.hidden_a]))
            hc += self.hidden_a
            out_collect.append(self.fc_grid_s_decode(h[:, hc: hc + self.hidden_s]))
            hc += self.hidden_s
            out_collect.append(self.fc_voxel_l_decode(h[:, hc: hc + self.hidden_vl]))
            hc += self.hidden_vl
            out_collect.append(self.fc_grid_p_decode(h[:, hc: hc + self.hidden_p]))
            hc += self.hidden_p
            out_collect.append(self.fc_voxel_p_decode(h[:, hc: hc + self.hidden_vp]))
        else:
            hc = 0
            out_collect.append(self.fc_grid_v_decode(h[:, hc: hc + self.hidden_v]))
            hc += self.hidden_v
            out_collect.append(self.fc_grid_a_decode(h[:, hc: hc + self.hidden_a]))
            hc += self.hidden_a
            out_collect.append(self.fc_grid_s_decode(h[:, hc: hc + self.hidden_s]))
            hc += self.hidden_s
            out_collect.append(self.fc_grid_p_decode(h[:, hc: hc + self.hidden_p]))

        return torch.cat(out_collect, dim=1)

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

# Loss function
def loss_function(recon_x, x, mean, logvar, last_n, last_n_weight,
                  kl_weight=1e-6):
    b = x.size(0) * x.size(1)
    recon_coords = F.l1_loss(recon_x[:, :, -last_n:], x[:, :, -last_n:], reduction="sum") / b
    recon_other = F.l1_loss(recon_x[:, :, :-last_n], x[:, :, :-last_n], reduction="sum") / b
    # KL divergence
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b
    return recon_coords * last_n_weight + recon_other + KLD * kl_weight

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
    vae = VAEVanilla(4, 139, 64, 128, 1)
    #print(vae)
    input_data = torch.randn(1, 4, 139)
    output_data, mean, logvar = vae(input_data)
    loss = loss_function(input_data, output_data, mean, logvar, 18, 40)
    loss.sum().backward() 
    for name, param in vae.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive gradients.")
