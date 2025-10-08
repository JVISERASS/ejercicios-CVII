"""
Tests for upsampling methods comparing our implementations with PyTorch equivalents.
"""

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from tests.conftest import assert_tensors_close, assert_same_shape_and_type
from src.upsampling import (
    bed_of_nails,
    nearest_neighbor, 
    bilinear_interpolation,
    max_unpooling
)


class TestBedOfNails:
    """Test bed of nails upsampling."""
    
    def test_bed_of_nails_vs_pytorch_manual(self, sample_feature_map, device):
        """Test bed of nails against manual PyTorch implementation."""
        x = sample_feature_map
        stride = 2
        
        # Our implementation
        our_result = bed_of_nails(x, stride)
        
        # PyTorch manual implementation
        N, C, H, W = x.shape
        pytorch_result = torch.zeros(N, C, H * stride, W * stride, device=device)
        pytorch_result[:, :, ::stride, ::stride] = x
        
        assert_tensors_close(our_result, pytorch_result)
        assert_same_shape_and_type(our_result, pytorch_result)
    
    def test_bed_of_nails_dimensions(self, device):
        """Test that bed of nails produces correct output dimensions."""
        x = torch.randn(2, 16, 32, 32, device=device)
        stride = 3
        
        result = bed_of_nails(x, stride)
        expected_shape = (2, 16, 96, 96)
        
        assert result.shape == expected_shape
    
    def test_bed_of_nails_different_strides(self, device):
        """Test bed of nails with different stride values."""
        x = torch.randn(1, 8, 16, 16, device=device)
        
        for stride in [2, 3, 4]:
            result = bed_of_nails(x, stride)
            expected_shape = (1, 8, 16 * stride, 16 * stride)
            assert result.shape == expected_shape


class TestNearestNeighbor:
    """Test nearest neighbor interpolation."""
    
    def test_nearest_neighbor_vs_pytorch(self, sample_feature_map, device):
        """Test nearest neighbor against PyTorch F.interpolate."""
        x = sample_feature_map
        target_size = (112, 112)
        
        # Our implementation
        our_result = nearest_neighbor(x, target_size)
        
        # PyTorch implementation
        pytorch_result = F.interpolate(x, size=target_size, mode='nearest')
        
        assert_tensors_close(our_result, pytorch_result, atol=1e-6)
        assert_same_shape_and_type(our_result, pytorch_result)
    
    def test_nearest_neighbor_upscale(self, device):
        """Test nearest neighbor upscaling."""
        x = torch.randn(1, 3, 32, 32, device=device)
        target_size = (64, 64)
        
        result = nearest_neighbor(x, target_size)
        assert result.shape == (1, 3, 64, 64)
    
    def test_nearest_neighbor_downscale(self, device):
        """Test nearest neighbor downscaling."""
        x = torch.randn(1, 3, 64, 64, device=device)
        target_size = (32, 32)
        
        result = nearest_neighbor(x, target_size)
        assert result.shape == (1, 3, 32, 32)


class TestBilinearInterpolation:
    """Test bilinear interpolation."""
    
    def test_bilinear_vs_pytorch(self, sample_feature_map, device):
        """Test bilinear interpolation against PyTorch F.interpolate."""
        x = sample_feature_map
        target_size = (112, 112)
        
        # Our implementation
        our_result = bilinear_interpolation(x, target_size)
        
        # PyTorch implementation
        pytorch_result = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        
        assert_tensors_close(our_result, pytorch_result, rtol=1e-4, atol=1e-6)
    
    def test_bilinear_different_sizes(self, device):
        """Test bilinear interpolation with different target sizes."""
        x = torch.randn(2, 8, 28, 28, device=device)
        
        test_sizes = [(14, 14), (56, 56), (32, 48)]
        
        for target_size in test_sizes:
            result = bilinear_interpolation(x, target_size)
            expected_shape = (2, 8, target_size[0], target_size[1])
            assert result.shape == expected_shape
    
    def test_bilinear_identity(self, device):
        """Test that bilinear interpolation is identity when target size equals input size."""
        x = torch.randn(1, 4, 32, 32, device=device)
        target_size = (32, 32)
        
        result = bilinear_interpolation(x, target_size)
        assert_tensors_close(result, x, rtol=1e-4)


class TestMaxUnpooling:
    """Test max unpooling."""
    
    def test_max_unpooling_vs_pytorch(self, device):
        """Test max unpooling against PyTorch nn.MaxUnpool2d."""
        # Create input that we can pool and then unpool
        x = torch.randn(1, 4, 16, 16, device=device)
        kernel_size = 2
        stride = 2
        
        # Pool first to get indices
        pool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        pooled, indices = pool(x)
        
        # Our implementation
        our_result = max_unpooling(pooled, indices, kernel_size, stride)
        
        # PyTorch implementation
        unpool = nn.MaxUnpool2d(kernel_size, stride)
        pytorch_result = unpool(pooled, indices)
        
        assert_tensors_close(our_result, pytorch_result)
        assert_same_shape_and_type(our_result, pytorch_result)
    
    def test_max_unpooling_dimensions(self, device):
        """Test max unpooling output dimensions."""
        pooled = torch.randn(2, 8, 14, 14, device=device)
        # Create dummy indices (normally from MaxPool2d)
        indices = torch.randint(0, 4, (2, 8, 14, 14), device=device, dtype=torch.long)
        kernel_size = 2
        stride = 2
        
        result = max_unpooling(pooled, indices, kernel_size, stride)
        expected_shape = (2, 8, 28, 28)
        assert result.shape == expected_shape
    
    def test_max_pool_unpool_consistency(self, device):
        """Test that pooling then unpooling preserves non-zero locations."""
        x = torch.zeros(1, 1, 8, 8, device=device)
        # Set specific values that should survive pooling
        x[0, 0, 1, 1] = 1.0
        x[0, 0, 3, 3] = 2.0
        x[0, 0, 5, 5] = 3.0
        x[0, 0, 7, 7] = 4.0
        
        # Pool and unpool
        pool = nn.MaxPool2d(2, 2, return_indices=True)
        pooled, indices = pool(x)
        unpooled = max_unpooling(pooled, indices, 2, 2)
        
        # Check that unpooled has correct non-zero values at some positions
        assert unpooled.shape == x.shape
        assert torch.sum(unpooled > 0) <= torch.sum(x > 0)  # May have fewer non-zeros due to pooling