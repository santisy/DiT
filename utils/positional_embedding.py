import numpy as np
import torch

def fourier_positional_encoding(locations: torch.Tensor, out_features):
    """
        Apply Fourier positional encoding to 3D locations with automatic frequency band calculation
        and optional zero-padding to match out_features.

        Parameters:
        - locations: Tensor of shape [N, L_i, 3] containing 3D locations.
        - out_features: The desired output dimensionality (C).

        Returns:
        - Tensor of shape [N, L_i, C] with Fourier positional encoding applied.
    """
    device = locations.device
    N, L_i, _ = locations.shape
    
    # Calculate num_freqs based on the desired output size (out_features)
    omega = torch.arange(out_features // 6, dtype=torch.float32)  # Each frequency generates 6 features (3 dimensions * sin and cos)
    omega /= out_features / 6
    
    # Generate frequency bands
    omega = 1. / 10000 ** omega
    
    # Prepare frequency bands for broadcasting
    omega = omega.view(1, 1, 1, -1).to(device)
    
    # Expand locations for multiplication with freq_bands
    locations_expanded = locations.unsqueeze(-1)
    
    # Scale locations by frequencies and apply sin and cos
    scaled_locations = locations_expanded * omega
    sin_features = torch.sin(scaled_locations).to(device)
    cos_features = torch.cos(scaled_locations).to(device)
    
    # Concatenate and reshape to [N, L_i, num_freqs * 6]
    encoded = torch.cat([sin_features, cos_features], dim=-1).reshape(N, L_i, -1)
    
    # Zero-pad if necessary to match out_features
    if encoded.shape[-1] < out_features:
        padding_size = out_features - encoded.shape[-1]
        encoded = torch.nn.functional.pad(encoded, (0, padding_size), "constant", 0)
    elif encoded.shape[-1] > out_features:
        encoded = encoded[..., :out_features]
    
    return encoded

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

    x_encoded = (x_encoded + 1.0) / 2.0

    return x_encoded
