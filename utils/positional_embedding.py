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
