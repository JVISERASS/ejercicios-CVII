"""
Point Clouds - Procesamiento de nubes de puntos 3D.
"""

import torch


def point_cloud_from_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Converts depth map to point cloud.
    
    Args:
        depth: Depth map [H, W]
        intrinsics: Camera intrinsic matrix [3, 3]
        
    Returns:
        Points [N, 3] - 3D point cloud
    """
    h, w = depth.shape
    
    # Create pixel coordinate grid
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), 
                          torch.arange(w, device=depth.device), 
                          indexing='ij')
    x = x.flatten().float()
    y = y.flatten().float()
    z = depth.flatten()
    
    # Extract camera parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Unproject to 3D coordinates
    X = (x - cx) * (z / fx)
    Y = (y - cy) * (z / fy)
    
    points = torch.stack([X, Y, z], dim=1)
    
    # Filter out invalid depth values
    valid_mask = z > 0
    points = points[valid_mask]
    
    return points


def downsample_point_cloud(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """
    Downsample point cloud using voxel grid filtering.
    
    Args:
        points: Point cloud [N, 3]
        voxel_size: Size of voxel for downsampling
        
    Returns:
        Downsampled point cloud [M, 3] where M <= N
    """
    # Quantize points to voxel grid
    voxel_coords = torch.floor(points / voxel_size).long()
    
    # Find unique voxel coordinates
    unique_voxels, inverse_indices = torch.unique(voxel_coords, return_inverse=True, dim=0)
    
    # For each unique voxel, take the centroid of points
    downsampled_points = torch.zeros(len(unique_voxels), 3, device=points.device)
    
    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        downsampled_points[i] = points[mask].mean(dim=0)
    
    return downsampled_points


def compute_point_normals(points: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Estimate surface normals for point cloud using local neighborhoods.
    
    Args:
        points: Point cloud [N, 3]
        k: Number of nearest neighbors to use
        
    Returns:
        Normals [N, 3] - estimated surface normals
    """
    N = points.shape[0]
    normals = torch.zeros_like(points)
    
    for i in range(N):
        # Find k nearest neighbors
        distances = torch.norm(points - points[i], dim=1)
        _, indices = torch.topk(distances, k + 1, largest=False)  # +1 to exclude self
        neighbors = points[indices[1:]]  # Exclude self
        
        # Center the neighborhood
        centered = neighbors - neighbors.mean(dim=0)
        
        # Compute covariance matrix and find principal component
        cov = torch.mm(centered.T, centered) / (k - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(cov)
        
        # Normal is the eigenvector corresponding to smallest eigenvalue
        min_idx = torch.argmin(eigenvalues.real)
        normal = eigenvectors[:, min_idx].real
        
        normals[i] = normal
    
    return normals