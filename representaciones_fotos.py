import os
from typing import Literal

import numpy as np
from skimage import segmentation, color
from skimage.future import graph
from skimage import io
from skimage.feature import selective_search 
ss = selective_search

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# Mapa de profundidad (Depth Map)
def depth_map(disparity: torch.Tensor, focal_length: float, baseline: float, epsilon: float) -> torch.Tensor:
    """
    Computes depth map from disparity.
    Args:
        disparity: Disparity tensor [H, W]
        focal_length: Camera focal length
        baseline: Distance between cameras
    Returns:
        Depth map tensor [H, W]
    """
    # si se esta trabajando con gpu pasar todo a tensor y al mismo device:
    # device = disparity.device
    # focal_length = torch.tensor(focal_length, device=device, dtype=disparity.dtype)
    # baseline = torch.tensor(baseline, device=device, dtype=disparity.dtype)
    # epsilon = torch.tensor(epsilon, device=device, dtype=disparity.dtype)
    return (focal_length * baseline) / (disparity + epsilon)

# Normal de superficie
def surface_normal(depth: torch.Tensor) -> torch.Tensor:
    """
    Estimates surface normals from depth map.
    Args:
        depth: Depth map [H, W]
    Returns:
        Normals [H, W, 3]
    """
    dist_xz = torch.gradient(depth, axis=1)[0]
    dist_yz = torch.gradient(depth, axis=0)[0]
    normal = torch.stack([-dist_xz, -dist_yz, torch.ones_like(depth)], dim=-1)
    normal = F.normalize(normal, dim=-1)
    return normal

# Funciones implícitas (Ejemplo: SDF)
def implicit_function_sphere(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Signed Distance Function for a sphere.
    Args:
        x: Points [N, 3]
        center: Center [3]
        radius: Radius
    Returns:
        SDF values [N]
    """
    return torch.norm(x - center, dim=1) - radius

# Voxels
def points_to_voxel(points: torch.Tensor, grid_size: tuple, bounds: tuple) -> torch.Tensor:
    """
    Converts point cloud to voxel grid.
    Args:
        points: [N, 3]
        grid_size: (X, Y, Z)
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    Returns:
        Voxel grid [X, Y, Z]
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
    
    voxels = torch.zeros(grid_size, dtype=torch.bool, device=points.device)
    voxels[idx[:,0], idx[:,1], idx[:,2]] = True

    return voxels
    
# Nube de puntos
def point_cloud_from_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Converts depth map to point cloud.
    Args:
        depth: [H, W]
        intrinsics: [3, 3] camera matrix
    Returns:
        Points [N, 3]
    """
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    x = x.flatten().float()
    y = y.flatten().float()
    z = depth.flatten()
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    X = (x - cx) * (z / fx)
    Y = (y - cy) * (z / fy)
    points = torch.stack([X,Y,z], dim=1)
   