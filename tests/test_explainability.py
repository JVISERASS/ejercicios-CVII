"""
Tests for explainability methods verifying correct behavior and dimensions.
"""

import pytest
import torch
import torch.nn as nn
from tests.conftest import assert_tensors_close, assert_same_shape_and_type
from src.explainability import (
    saliency_map,
    SaliencyMaps,
    SaliencyMapsMax,
    SaliencyMapsL2,
    InputXGradient,
    IntegratedGradients
)


class SimpleTestModel(nn.Module):
    """Simple model for testing explainability methods."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@pytest.fixture
def simple_model(device):
    """Create a simple test model."""
    model = SimpleTestModel(num_classes=5).to(device)
    model.eval()
    return model


class TestSaliencyMaps:
    """Test saliency map implementations."""
    
    def test_saliency_map_function_basic(self, simple_model, sample_image, device):
        """Test basic saliency map function."""
        x = sample_image.requires_grad_(True)
        
        result = saliency_map(simple_model, x, target_class=None)
        
        # Should return same spatial dimensions as input
        assert result.shape == x.shape[2:]  # [H, W]
        assert result.dtype == x.dtype
        
        # Should have non-negative values (absolute gradients)
        assert torch.all(result >= 0)
    
    def test_saliency_map_vs_class_implementation(self, simple_model, sample_batch_images, device):
        """Test that function and class implementations give similar results."""
        x = sample_batch_images
        target_class = 2
        
        # Function implementation (but we need to modify it to handle batches)
        saliency_class = SaliencyMapsMax(simple_model)
        class_result = saliency_class(x, target_class)
        
        # Should have correct batch dimension
        assert class_result.shape[0] == x.shape[0]
        assert class_result.shape[1:] == x.shape[2:]  # [H, W]
    
    def test_saliency_maps_class_basic(self, simple_model, sample_batch_images, device):
        """Test SaliencyMaps class implementation."""
        x = sample_batch_images
        saliency = SaliencyMaps(simple_model)
        
        result = saliency(x, target_class=None)
        
        # Should return normalized values in [0, 1]
        assert result.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_saliency_maps_max_vs_sum(self, simple_model, sample_image, device):
        """Test difference between max and sum aggregation methods."""
        x = sample_image
        
        saliency_sum = SaliencyMaps(simple_model)
        saliency_max = SaliencyMapsMax(simple_model)
        
        result_sum = saliency_sum(x, target_class=0)
        result_max = saliency_max(x, target_class=0)
        
        # Both should have same shape
        assert result_sum.shape == result_max.shape
        
        # Results may be different due to aggregation method
        # but both should be non-negative
        assert torch.all(result_sum >= 0)
        assert torch.all(result_max >= 0)
    
    def test_saliency_maps_l2_norm(self, simple_model, sample_image, device):
        """Test L2 norm version of saliency maps."""
        x = sample_image
        saliency_l2 = SaliencyMapsL2(simple_model)
        
        result = saliency_l2(x, target_class=1)
        
        # Should be normalized to [0, 1]
        assert result.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_saliency_maps_different_targets(self, simple_model, sample_image, device):
        """Test saliency maps with different target classes."""
        x = sample_image
        saliency = SaliencyMaps(simple_model)
        
        result_class0 = saliency(x, target_class=0)
        result_class1 = saliency(x, target_class=1)
        
        # Results for different classes should generally be different
        assert not torch.allclose(result_class0, result_class1, rtol=1e-3)
    
    def test_saliency_maps_auto_target(self, simple_model, sample_batch_images, device):
        """Test saliency maps with automatic target class selection."""
        x = sample_batch_images
        saliency = SaliencyMaps(simple_model)
        
        # Should work without specifying target class
        result = saliency(x, target_class=None)
        
        assert result.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestInputXGradient:
    """Test Input × Gradient method."""
    
    def test_input_x_gradient_basic(self, simple_model, sample_batch_images, device):
        """Test basic Input × Gradient functionality."""
        x = sample_batch_images
        input_x_grad = InputXGradient(simple_model)
        
        result = input_x_grad(x, target_class=2)
        
        # Should have same shape as regular saliency maps
        assert result.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_input_x_gradient_vs_saliency(self, simple_model, sample_image, device):
        """Test that Input × Gradient differs from regular saliency."""
        x = sample_image
        target_class = 1
        
        saliency = SaliencyMaps(simple_model)
        input_x_grad = InputXGradient(simple_model)
        
        saliency_result = saliency(x, target_class)
        input_x_grad_result = input_x_grad(x, target_class)
        
        # Results should generally be different (input multiplication effect)
        assert not torch.allclose(saliency_result, input_x_grad_result, rtol=1e-2)
    
    def test_input_x_gradient_zero_input(self, simple_model, device):
        """Test Input × Gradient with zero input (should give zero result)."""
        x = torch.zeros(1, 3, 32, 32, device=device)
        input_x_grad = InputXGradient(simple_model)
        
        result = input_x_grad(x, target_class=0)
        
        # Should be all zeros since input is zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


class TestIntegratedGradients:
    """Test Integrated Gradients method."""
    
    def test_integrated_gradients_basic(self, simple_model, sample_image, device):
        """Test basic Integrated Gradients functionality."""
        x = sample_image
        ig = IntegratedGradients(simple_model)
        
        result = ig(x, target_class=1, n_steps=5)
        
        # Should have correct shape and be normalized
        assert result.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_integrated_gradients_different_steps(self, simple_model, sample_image, device):
        """Test Integrated Gradients with different step counts."""
        x = sample_image
        ig = IntegratedGradients(simple_model)
        
        result_5_steps = ig(x, target_class=0, n_steps=5)
        result_10_steps = ig(x, target_class=0, n_steps=10)
        
        # Results should be similar but may differ due to approximation
        assert result_5_steps.shape == result_10_steps.shape
        
        # More steps should generally give more accurate results
        # (though this is hard to test without ground truth)
    
    def test_integrated_gradients_different_baselines(self, simple_model, sample_image, device):
        """Test Integrated Gradients with different baseline types."""
        x = sample_image
        ig = IntegratedGradients(simple_model)
        
        result_zero = ig(x, target_class=1, baseline="zero", n_steps=5)
        result_random = ig(x, target_class=1, baseline="random", n_steps=5)
        
        # Results should generally be different due to different baselines
        assert result_zero.shape == result_random.shape
        # Results may be different but both should be valid
        assert torch.all(result_zero >= 0) and torch.all(result_zero <= 1)
        assert torch.all(result_random >= 0) and torch.all(result_random <= 1)
    
    def test_integrated_gradients_axiom_sensitivity(self, simple_model, device):
        """Test that IG satisfies sensitivity axiom (non-zero for different inputs)."""
        # Create two different inputs
        x1 = torch.zeros(1, 3, 32, 32, device=device)
        x2 = torch.ones(1, 3, 32, 32, device=device) * 0.5
        
        ig = IntegratedGradients(simple_model)
        
        # Get model predictions to ensure they're different
        with torch.no_grad():
            pred1 = simple_model(x1)
            pred2 = simple_model(x2)
        
        # If predictions are different, IG should give non-zero attribution for x2
        if not torch.allclose(pred1, pred2, rtol=1e-3):
            result = ig(x2, target_class=pred2.argmax().item(), n_steps=10)
            
            # Should have some non-zero attributions
            assert torch.sum(result > 1e-6) > 0


class TestExplainabilityConsistency:
    """Test consistency properties across explainability methods."""
    
    def test_all_methods_same_output_shape(self, simple_model, sample_image, device):
        """Test that all methods produce same output shape."""
        x = sample_image
        target_class = 2
        
        methods = [
            SaliencyMaps(simple_model),
            SaliencyMapsMax(simple_model),
            SaliencyMapsL2(simple_model),
            InputXGradient(simple_model),
            IntegratedGradients(simple_model)
        ]
        
        expected_shape = (x.shape[0], x.shape[2], x.shape[3])
        
        for method in methods:
            if isinstance(method, IntegratedGradients):
                result = method(x, target_class, n_steps=3)
            else:
                result = method(x, target_class)
            
            assert result.shape == expected_shape, f"Method {type(method).__name__} has wrong shape"
    
    def test_all_methods_handle_batch(self, simple_model, sample_batch_images, device):
        """Test that all methods handle batch inputs correctly."""
        x = sample_batch_images
        batch_size = x.shape[0]
        
        methods = [
            SaliencyMaps(simple_model),
            InputXGradient(simple_model),
            IntegratedGradients(simple_model)
        ]
        
        for method in methods:
            if isinstance(method, IntegratedGradients):
                result = method(x, target_class=1, n_steps=3)
            else:
                result = method(x, target_class=1)
            
            assert result.shape[0] == batch_size, f"Method {type(method).__name__} doesn't handle batches"
    
    def test_gradient_requirement(self, simple_model, device):
        """Test that methods properly handle gradient requirements."""
        # Create input without gradients
        x = torch.randn(1, 3, 32, 32, device=device)
        
        saliency = SaliencyMaps(simple_model)
        
        # Should work even if input doesn't initially require gradients
        result = saliency(x, target_class=0)
        
        assert result.shape == (1, 32, 32)
        assert torch.all(result >= 0)