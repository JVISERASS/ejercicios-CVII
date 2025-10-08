"""
Depth Maps - Cálculo de mapas de profundidad a partir de disparidad.
"""

import torch


def depth_map(disparity: torch.Tensor, focal_length: float, baseline: float, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes depth map from disparity.
    
    Args:
        disparity: Disparity tensor [H, W]
        focal_length: Camera focal length
        baseline: Distance between cameras
        epsilon: Small value to avoid division by zero
        
    Returns:
        Depth map tensor [H, W]
    """
    # Convert parameters to tensors if working on GPU
    device = disparity.device
    focal_length = torch.tensor(focal_length, device=device, dtype=disparity.dtype)
    baseline = torch.tensor(baseline, device=device, dtype=disparity.dtype)
    epsilon = torch.tensor(epsilon, device=device, dtype=disparity.dtype)
    
    return (focal_length * baseline) / (disparity + epsilon)