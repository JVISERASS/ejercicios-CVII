"""
Surface Normals - Estimación de normales de superficie a partir de mapas de profundidad.
"""

import torch
import torch.nn.functional as F


def surface_normal(depth: torch.Tensor) -> torch.Tensor:
    """
    Estimates surface normals from depth map.
    
    Args:
        depth: Depth map [H, W]
        
    Returns:
        Normals [H, W, 3] - normalized surface normal vectors
    """
    # Calculate gradients in x and y directions
    dist_xz = torch.gradient(depth, dim=1)[0]
    dist_yz = torch.gradient(depth, dim=0)[0]
    
    # Construct normal vectors
    normal = torch.stack([-dist_xz, -dist_yz, torch.ones_like(depth)], dim=-1)
    
    # Normalize to unit vectors
    normal = F.normalize(normal, dim=-1)
    
    return normal