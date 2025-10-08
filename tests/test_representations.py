"""
Tests for 3D representations comparing with manual calculations and expected behaviors.
"""

import pytest
import torch
import numpy as np
from tests.conftest import assert_tensors_close, assert_same_shape_and_type
from src.representations import (
    depth_map,
    surface_normal,
    points_to_voxel,
    voxel_to_points,
    point_cloud_from_depth
)


class TestDepthMap:
    """Test depth map calculation from disparity."""
    
    def test_depth_map_basic(self, device):
        """Test basic depth map calculation."""
        disparity = torch.tensor([[1.0, 2.0], [0.5, 4.0]], device=device)
        focal_length = 525.0
        baseline = 0.1
        epsilon = 1e-8
        
        result = depth_map(disparity, focal_length, baseline, epsilon)
        
        # Manual calculation: depth = (focal_length * baseline) / disparity
        expected = torch.tensor([
            [525.0 * 0.1 / 1.0, 525.0 * 0.1 / 2.0],
            [525.0 * 0.1 / 0.5, 525.0 * 0.1 / 4.0]
        ], device=device)
        
        assert_tensors_close(result, expected, rtol=1e-5)
    
    def test_depth_map_zero_disparity(self, device):
        """Test depth map with zero disparity (should not crash due to epsilon)."""
        disparity = torch.tensor([[0.0, 1.0]], device=device)
        focal_length = 525.0
        baseline = 0.1
        epsilon = 1e-8
        
        result = depth_map(disparity, focal_length, baseline, epsilon)
        
        # Should not be infinite due to epsilon
        assert torch.all(torch.isfinite(result))
        assert result[0, 0] > 0  # Should be very large but finite
    
    def test_depth_map_different_parameters(self, device):
        """Test depth map with different focal length and baseline."""
        disparity = torch.ones(3, 3, device=device)
        
        # Test different focal lengths
        result1 = depth_map(disparity, 100.0, 0.1, 1e-8)
        result2 = depth_map(disparity, 200.0, 0.1, 1e-8)
        
        # Doubling focal length should double the depth
        assert_tensors_close(result2, 2 * result1, rtol=1e-5)
        
        # Test different baselines
        result3 = depth_map(disparity, 100.0, 0.2, 1e-8)
        
        # Doubling baseline should double the depth
        assert_tensors_close(result3, 2 * result1, rtol=1e-5)


class TestSurfaceNormal:
    """Test surface normal estimation from depth maps."""
    
    def test_surface_normal_flat_surface(self, device):
        """Test surface normals for a flat surface (should point up)."""
        # Flat surface with constant depth
        depth = torch.ones(10, 10, device=device) * 5.0
        
        result = surface_normal(depth)
        
        # Gradients should be zero, normal should be (0, 0, 1) everywhere
        expected = torch.zeros(10, 10, 3, device=device)
        expected[:, :, 2] = 1.0  # Z component should be 1
        
        assert_tensors_close(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_surface_normal_sloped_surface(self, device):
        """Test surface normals for a sloped surface."""
        # Create a surface that slopes in X direction
        x = torch.arange(5, device=device, dtype=torch.float32)
        y = torch.arange(5, device=device, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Depth increases linearly with x
        depth = xx.float()
        
        result = surface_normal(depth)
        
        # Should have consistent normal directions (negative X component)
        # All normals should be normalized
        norms = torch.norm(result, dim=-1)
        assert_tensors_close(norms, torch.ones_like(norms), rtol=1e-4)
        
        # X component should be negative (surface slopes up in X)
        assert torch.all(result[:, :, 0] < 0)
    
    def test_surface_normal_dimensions(self, device):
        """Test that surface normal output has correct dimensions."""
        depth = torch.randn(32, 64, device=device)
        
        result = surface_normal(depth)
        
        assert result.shape == (32, 64, 3)
        
        # Check that all normals are unit vectors
        norms = torch.norm(result, dim=-1)
        assert_tensors_close(norms, torch.ones_like(norms), rtol=1e-4)


class TestVoxelOperations:
    """Test voxel grid operations."""
    
    def test_points_to_voxel_basic(self, device):
        """Test basic point cloud to voxel conversion."""
        # Simple points at corners of unit cube
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        grid_size = (2, 2, 2)
        bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        
        result = points_to_voxel(points, grid_size, bounds)
        
        # Should have occupied voxels at expected positions
        assert result.shape == grid_size
        assert result.dtype == torch.bool
        assert torch.sum(result) >= 1  # At least one voxel should be occupied
    
    def test_voxel_to_points_basic(self, device):
        """Test voxel grid to point cloud conversion."""
        grid_size = (4, 4, 4)
        voxels = torch.zeros(grid_size, dtype=torch.bool, device=device)
        
        # Set some voxels as occupied
        voxels[0, 0, 0] = True
        voxels[2, 2, 2] = True
        voxels[3, 1, 3] = True
        
        bounds = ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))
        
        result = voxel_to_points(voxels, bounds)
        
        # Should return points for each occupied voxel
        assert result.shape[0] == 3  # 3 occupied voxels
        assert result.shape[1] == 3  # 3D coordinates
    
    def test_points_voxel_roundtrip(self, device):
        """Test that points->voxel->points preserves general structure."""
        # Create points in a grid pattern
        points = torch.tensor([
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
        ], device=device)
        
        grid_size = (4, 4, 4)
        bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        
        # Convert to voxel and back
        voxels = points_to_voxel(points, grid_size, bounds)
        reconstructed_points = voxel_to_points(voxels, bounds)
        
        # Should have some points (may be different due to quantization)
        assert reconstructed_points.shape[0] > 0
        assert reconstructed_points.shape[1] == 3
        
        # All reconstructed points should be within bounds
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        
        assert torch.all(reconstructed_points[:, 0] >= x_min)
        assert torch.all(reconstructed_points[:, 0] <= x_max)
        assert torch.all(reconstructed_points[:, 1] >= y_min)
        assert torch.all(reconstructed_points[:, 1] <= y_max)
        assert torch.all(reconstructed_points[:, 2] >= z_min)
        assert torch.all(reconstructed_points[:, 2] <= z_max)


class TestPointCloudFromDepth:
    """Test point cloud generation from depth maps."""
    
    def test_point_cloud_from_depth_basic(self, sample_camera_intrinsics, device):
        """Test basic point cloud generation from depth map."""
        # Simple 3x3 depth map
        depth = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0]
        ], device=device)
        
        result = point_cloud_from_depth(depth, sample_camera_intrinsics)
        
        # Should have 9 points (3x3) with valid depth
        assert result.shape == (9, 3)
        
        # All Z coordinates should match depth values
        expected_z = depth.flatten()
        assert_tensors_close(result[:, 2], expected_z, rtol=1e-5)
    
    def test_point_cloud_dimensions(self, sample_camera_intrinsics, device):
        """Test point cloud dimensions for different depth map sizes."""
        for h, w in [(10, 15), (32, 32), (64, 48)]:
            depth = torch.ones(h, w, device=device)
            
            result = point_cloud_from_depth(depth, sample_camera_intrinsics)
            
            assert result.shape == (h * w, 3)
    
    def test_point_cloud_camera_center(self, device):
        """Test that center pixel projects to correct 3D point."""
        # Create depth map where only center has depth
        depth = torch.zeros(5, 5, device=device)
        depth[2, 2] = 1.0  # Center pixel at depth 1
        
        # Camera with center at (2, 2)
        intrinsics = torch.tensor([
            [1.0, 0.0, 2.0],  # cx = 2
            [0.0, 1.0, 2.0],  # cy = 2
            [0.0, 0.0, 1.0]
        ], device=device)
        
        result = point_cloud_from_depth(depth, intrinsics)
        
        # Center pixel should project to (0, 0, 1) in 3D
        center_point_idx = 2 * 5 + 2  # Index of center pixel in flattened array
        center_3d = result[center_point_idx]
        
        expected = torch.tensor([0.0, 0.0, 1.0], device=device)
        assert_tensors_close(center_3d, expected, rtol=1e-5)
    
    def test_point_cloud_invalid_depth(self, sample_camera_intrinsics, device):
        """Test point cloud generation with some invalid (zero) depth values."""
        depth = torch.tensor([
            [1.0, 0.0, 1.0],  # Middle pixel has invalid depth
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]   # Corner pixels have invalid depth
        ], device=device)
        
        result = point_cloud_from_depth(depth, sample_camera_intrinsics)
        
        # Should still return all points, but some will have zero Z
        assert result.shape == (9, 3)
        
        # Points with zero depth should have Z coordinate of 0
        zero_depth_mask = depth.flatten() == 0
        assert torch.all(result[zero_depth_mask, 2] == 0)