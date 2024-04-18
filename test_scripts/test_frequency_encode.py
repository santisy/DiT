import torch
import numpy as np

def positional_encode(x, max_power):
    """
    Apply positional encoding as used in NeRF to a tensor with an additional token dimension.

    Args:
    x (torch.Tensor): Input tensor of shape (B, T, L) where each element is a scalar coordinate value.
    max_power (int): The maximum power of two to use in the encoding (exclusive).

    Returns:
    torch.Tensor: Encoded tensor of shape (B, T, C) where C = L * 2 * max_power.
    """
    B, T, L = x.shape
    x = x.unsqueeze(-1)  # Shape: (B, T, L, 1)
    powers = torch.tensor([2**i for i in range(max_power)]).to(x.device, x.dtype)
    x_scaled = x * powers  # Shape: (B, T, L, max_power)
    
    # Apply sine and cosine to scaled x
    sin_x = torch.sin(2 * np.pi * x_scaled)
    cos_x = torch.cos(2 * np.pi * x_scaled)
    
    # Concatenate along the last dimension and then flatten across the last two dimensions
    x_encoded = torch.cat([sin_x, cos_x], dim=-1)  # Shape: (B, T, L, 2 * max_power)
    x_encoded = x_encoded.flatten(start_dim=2)  # Shape: (B, T, L * 2 * max_power)

    return x_encoded


if __name__ == "__main__":
    a = torch.randn(16, 8, 3)
    pe = positional_encode(a, 16)
    print(pe.shape)
