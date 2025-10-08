"""
Voxels - Conversión de nubes de puntos a grillas voxelizadas.
"""

import torch


def points_to_voxel(points: torch.Tensor, grid_size: tuple, bounds: tuple) -> torch.Tensor:
    """
    Converts point cloud to voxel grid.
    
    Args:
        points: Point cloud [N, 3]
        grid_size: (X, Y, Z) - dimensions of voxel grid
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) - spatial bounds
        
    Returns:
        Voxel grid [X, Y, Z] - boolean tensor indicating occupied voxels
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    # Shift points to origin
    shifted = points - torch.tensor([x_min, y_min, z_min], device=points.device)
    
    # Normalize to [0, 1]
    size = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=points.device)
    normalized = shifted / size
    
    # Scale to grid size
    scaled = normalized * torch.tensor(grid_size, device=points.device)
    
    # Convert to integer indices
    idx = scaled.long()
    
    # Clamp indices to grid bounds
    max_idx = torch.tensor(grid_size, device=points.device) - 1
    idx = torch.clamp(idx, min=0, max=max_idx)
    
    # Create voxel grid
    voxels = torch.zeros(grid_size, dtype=torch.bool, device=points.device)
    voxels[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return voxels


def voxel_to_points(voxels: torch.Tensor, bounds: tuple) -> torch.Tensor:
    """
    Converts voxel grid back to point cloud.
    
    Args:
        voxels: Voxel grid [X, Y, Z] - boolean tensor
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) - spatial bounds
        
    Returns:
        Point cloud [N, 3] - points at voxel centers
    """
    # Get indices of occupied voxels
    occupied = torch.nonzero(voxels, as_tuple=False).float()
    
    # Convert back to world coordinates
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]
    
    grid_size = torch.tensor(voxels.shape, device=voxels.device, dtype=torch.float32)
    size = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=voxels.device)
    
    # Normalize from grid coordinates to [0, 1]
    normalized = occupied / grid_size
    
    # Scale to world coordinates
    points = normalized * size + torch.tensor([x_min, y_min, z_min], device=voxels.device)
    
    return points