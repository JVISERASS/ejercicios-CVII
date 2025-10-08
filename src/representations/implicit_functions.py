"""
Implicit Functions - Funciones implícitas como Signed Distance Functions (SDF).
"""

import torch


def implicit_function_sphere(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Signed Distance Function for a sphere.
    
    Args:
        x: Points [N, 3] - query points in 3D space
        center: Center [3] - sphere center coordinates
        radius: Radius of the sphere
        
    Returns:
        SDF values [N] - signed distances to sphere surface
    """
    distances = torch.norm(x - center, dim=1)
    return distances - radius


def implicit_function_box(x: torch.Tensor, center: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """
    Signed Distance Function for an axis-aligned box.
    
    Args:
        x: Points [N, 3] - query points in 3D space
        center: Center [3] - box center coordinates  
        size: Size [3] - box dimensions (width, height, depth)
        
    Returns:
        SDF values [N] - signed distances to box surface
    """
    # Translate to box-centered coordinates
    q = torch.abs(x - center) - size / 2
    
    # Distance to box surface
    outside = torch.norm(torch.clamp(q, min=0), dim=1)
    inside = torch.clamp(torch.max(q, dim=1)[0], max=0)
    
    return outside + inside


def implicit_function_plane(x: torch.Tensor, normal: torch.Tensor, distance: float) -> torch.Tensor:
    """
    Signed Distance Function for a plane.
    
    Args:
        x: Points [N, 3] - query points in 3D space
        normal: Normal [3] - normalized plane normal vector
        distance: Distance from origin to plane
        
    Returns:
        SDF values [N] - signed distances to plane
    """
    return torch.dot(x, normal.unsqueeze(0)) - distance