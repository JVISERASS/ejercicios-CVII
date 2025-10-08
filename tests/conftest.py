"""
Tests configuration and fixtures for the computer vision exercises.
"""

import pytest
import torch
import numpy as np
from typing import Generator


@pytest.fixture
def device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image(device: torch.device) -> torch.Tensor:
    """Create a sample image tensor [1, 3, 224, 224]."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224, device=device)


@pytest.fixture
def sample_batch_images(device: torch.device) -> torch.Tensor:
    """Create a batch of sample images [4, 3, 224, 224]."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224, device=device)


@pytest.fixture
def sample_feature_map(device: torch.device) -> torch.Tensor:
    """Create a sample feature map [1, 64, 56, 56]."""
    torch.manual_seed(42)
    return torch.randn(1, 64, 56, 56, device=device)


@pytest.fixture
def sample_depth_map(device: torch.device) -> torch.Tensor:
    """Create a sample depth map [224, 224]."""
    torch.manual_seed(42)
    # Avoid zero depth values
    return torch.rand(224, 224, device=device) * 10 + 0.1


@pytest.fixture
def sample_point_cloud(device: torch.device) -> torch.Tensor:
    """Create a sample point cloud [1000, 3]."""
    torch.manual_seed(42)
    return torch.randn(1000, 3, device=device)


@pytest.fixture
def sample_boxes(device: torch.device) -> torch.Tensor:
    """Create sample bounding boxes [4, 4] in format (x, y, w, h)."""
    torch.manual_seed(42)
    return torch.tensor([
        [50, 50, 100, 100],
        [30, 30, 80, 80],
        [200, 200, 50, 50],
        [150, 150, 100, 80]
    ], dtype=torch.float32, device=device)


@pytest.fixture
def sample_camera_intrinsics(device: torch.device) -> torch.Tensor:
    """Create sample camera intrinsics matrix [3, 3]."""
    return torch.tensor([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)


def assert_tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                        rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """Assert that two tensors are close in value."""
    assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
        f"Tensors not close enough. Max diff: {torch.max(torch.abs(tensor1 - tensor2))}"


def assert_same_shape_and_type(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """Assert that two tensors have same shape and dtype."""
    assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    assert tensor1.dtype == tensor2.dtype, f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}"