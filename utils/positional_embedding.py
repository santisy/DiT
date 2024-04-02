import torch

def fourier_positional_encoding(locations, out_features):
    """
        Apply Fourier positional encoding to 3D locations with automatic frequency band calculation
        and optional zero-padding to match out_features.

        Parameters:
        - locations: Tensor of shape [N, L_i, 3] containing 3D locations.
        - out_features: The desired output dimensionality (C).

        Returns:
        - Tensor of shape [N, L_i, C] with Fourier positional encoding applied.
    """
    N, L_i, _ = locations.shape
    
    # Calculate num_freqs based on the desired output size (out_features)
    num_freqs = out_features // 6  # Each frequency generates 6 features (3 dimensions * sin and cos)
    
    # Generate frequency bands
    freq_bands = 2.0 ** torch.arange(num_freqs)
    
    # Prepare frequency bands for broadcasting
    freq_bands = freq_bands.view(1, 1, 1, -1)
    
    # Expand locations for multiplication with freq_bands
    locations_expanded = locations.unsqueeze(-1)
    
    # Scale locations by frequencies and apply sin and cos
    scaled_locations = locations_expanded * freq_bands
    sin_features = torch.sin(scaled_locations)
    cos_features = torch.cos(scaled_locations)
    
    # Concatenate and reshape to [N, L_i, num_freqs * 6]
    encoded = torch.cat([sin_features, cos_features], dim=-1).reshape(N, L_i, -1)
    
    # Zero-pad if necessary to match out_features
    if encoded.shape[-1] < out_features:
        padding_size = out_features - encoded.shape[-1]
        encoded = torch.nn.functional.pad(encoded, (0, padding_size), "constant", 0)
    elif encoded.shape[-1] > out_features:
        encoded = encoded[..., :out_features]
    
    return encoded