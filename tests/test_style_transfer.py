"""
Tests for style transfer losses and methods.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.conftest import assert_tensors_close, assert_same_shape_and_type
from src.style_transfer import (
    ContentLoss,
    StyleLoss,
    TotalVariationLoss,
    PerceptualLoss,
    GramMatrix,
    FeatureExtractor
)


class SimpleFeatureExtractor(nn.Module):
    """Simple feature extractor for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    
    def forward(self, x):
        features = {}
        x = F.relu(self.conv1(x))
        features['layer1'] = x
        x = F.relu(self.conv2(x))
        features['layer2'] = x
        x = F.relu(self.conv3(x))
        features['layer3'] = x
        return features


@pytest.fixture
def feature_extractor(device):
    """Create a simple feature extractor for testing."""
    return SimpleFeatureExtractor().to(device)


class TestGramMatrix:
    """Test Gram matrix computation."""
    
    def test_gram_matrix_basic(self, device):
        """Test basic Gram matrix computation."""
        gram = GramMatrix()
        
        # Create test feature maps [batch, channels, height, width]
        features = torch.randn(2, 64, 16, 16, device=device)
        
        result = gram(features)
        
        # Gram matrix should be [batch, channels, channels]
        assert result.shape == (2, 64, 64)
        
        # Gram matrix should be symmetric
        for i in range(result.shape[0]):
            gram_matrix = result[i]
            assert torch.allclose(gram_matrix, gram_matrix.T, rtol=1e-5)
    
    def test_gram_matrix_vs_manual_computation(self, device):
        """Compare Gram matrix with manual computation."""
        gram = GramMatrix()
        features = torch.randn(1, 32, 8, 8, device=device)
        
        # Our implementation
        result = gram(features)
        
        # Manual computation
        b, c, h, w = features.shape
        features_flat = features.view(b, c, h * w)  # [batch, channels, hw]
        manual_gram = torch.bmm(features_flat, features_flat.transpose(1, 2))  # [batch, c, c]
        manual_gram = manual_gram / (h * w)  # Normalize by spatial size
        
        assert_tensors_close(result, manual_gram, rtol=1e-5)
    
    def test_gram_matrix_normalization(self, device):
        """Test that Gram matrix is properly normalized."""
        gram = GramMatrix()
        
        # Test with different spatial sizes
        for size in [4, 8, 16]:
            features = torch.ones(1, 10, size, size, device=device)
            result = gram(features)
            
            # For constant input, Gram matrix elements should equal 1.0
            expected = torch.ones(1, 10, 10, device=device)
            assert_tensors_close(result, expected, rtol=1e-5)


class TestContentLoss:
    """Test content loss implementation."""
    
    def test_content_loss_basic(self, device):
        """Test basic content loss computation."""
        content_loss = ContentLoss()
        
        target_features = torch.randn(2, 64, 32, 32, device=device)
        input_features = torch.randn(2, 64, 32, 32, device=device)
        
        loss = content_loss(input_features, target_features)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_content_loss_vs_mse(self, device):
        """Compare content loss with MSE loss."""
        content_loss = ContentLoss()
        mse_loss = nn.MSELoss()
        
        target = torch.randn(2, 32, 16, 16, device=device)
        input_tensor = torch.randn(2, 32, 16, 16, device=device)
        
        our_loss = content_loss(input_tensor, target)
        pytorch_loss = mse_loss(input_tensor, target)
        
        # Should be identical to MSE loss
        assert_tensors_close(our_loss, pytorch_loss, rtol=1e-6)
    
    def test_content_loss_identical_inputs(self, device):
        """Test content loss with identical inputs."""
        content_loss = ContentLoss()
        
        features = torch.randn(1, 64, 16, 16, device=device)
        loss = content_loss(features, features)
        
        # Loss should be zero for identical inputs
        assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


class TestStyleLoss:
    """Test style loss implementation."""
    
    def test_style_loss_basic(self, device):
        """Test basic style loss computation."""
        style_loss = StyleLoss()
        
        target_features = torch.randn(2, 64, 16, 16, device=device)
        input_features = torch.randn(2, 64, 16, 16, device=device)
        
        loss = style_loss(input_features, target_features)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_style_loss_vs_gram_mse(self, device):
        """Compare style loss with manual Gram matrix + MSE."""
        style_loss = StyleLoss()
        gram = GramMatrix()
        mse = nn.MSELoss()
        
        target = torch.randn(1, 32, 8, 8, device=device)
        input_tensor = torch.randn(1, 32, 8, 8, device=device)
        
        our_loss = style_loss(input_tensor, target)
        
        # Manual computation
        target_gram = gram(target)
        input_gram = gram(input_tensor)
        manual_loss = mse(input_gram, target_gram)
        
        assert_tensors_close(our_loss, manual_loss, rtol=1e-5)
    
    def test_style_loss_identical_inputs(self, device):
        """Test style loss with identical inputs."""
        style_loss = StyleLoss()
        
        features = torch.randn(1, 48, 12, 12, device=device)
        loss = style_loss(features, features)
        
        # Loss should be zero for identical inputs
        assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


class TestTotalVariationLoss:
    """Test total variation loss implementation."""
    
    def test_tv_loss_basic(self, device):
        """Test basic total variation loss."""
        tv_loss = TotalVariationLoss()
        
        image = torch.randn(2, 3, 32, 32, device=device)
        loss = tv_loss(image)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_tv_loss_smooth_image(self, device):
        """Test TV loss on smooth image (should be low)."""
        tv_loss = TotalVariationLoss()
        
        # Create smooth image (constant values)
        smooth_image = torch.ones(1, 3, 16, 16, device=device)
        smooth_loss = tv_loss(smooth_image)
        
        # Create noisy image
        noisy_image = torch.randn(1, 3, 16, 16, device=device)
        noisy_loss = tv_loss(noisy_image)
        
        # Smooth image should have lower TV loss
        assert smooth_loss < noisy_loss
    
    def test_tv_loss_vs_manual_computation(self, device):
        """Compare TV loss with manual computation."""
        tv_loss = TotalVariationLoss()
        image = torch.randn(1, 3, 8, 8, device=device)
        
        our_loss = tv_loss(image)
        
        # Manual computation
        diff_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        diff_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        manual_loss = torch.mean(diff_x) + torch.mean(diff_y)
        
        assert_tensors_close(our_loss, manual_loss, rtol=1e-5)
    
    def test_tv_loss_different_weights(self, device):
        """Test TV loss with different weights."""
        image = torch.randn(1, 3, 16, 16, device=device)
        
        tv_loss_1 = TotalVariationLoss(weight=1.0)
        tv_loss_2 = TotalVariationLoss(weight=2.0)
        
        loss_1 = tv_loss_1(image)
        loss_2 = tv_loss_2(image)
        
        # Loss with weight 2.0 should be double
        assert_tensors_close(loss_2, 2.0 * loss_1, rtol=1e-5)


class TestPerceptualLoss:
    """Test perceptual loss implementation."""
    
    def test_perceptual_loss_basic(self, feature_extractor, device):
        """Test basic perceptual loss computation."""
        perceptual_loss = PerceptualLoss(feature_extractor, ['layer1', 'layer2'])
        
        image1 = torch.randn(1, 3, 32, 32, device=device)
        image2 = torch.randn(1, 3, 32, 32, device=device)
        
        loss = perceptual_loss(image1, image2)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_perceptual_loss_identical_images(self, feature_extractor, device):
        """Test perceptual loss with identical images."""
        perceptual_loss = PerceptualLoss(feature_extractor, ['layer1'])
        
        image = torch.randn(1, 3, 32, 32, device=device)
        loss = perceptual_loss(image, image)
        
        # Loss should be zero for identical images
        assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-6)
    
    def test_perceptual_loss_multiple_layers(self, feature_extractor, device):
        """Test perceptual loss with multiple layers."""
        single_layer = PerceptualLoss(feature_extractor, ['layer1'])
        multi_layer = PerceptualLoss(feature_extractor, ['layer1', 'layer2', 'layer3'])
        
        image1 = torch.randn(1, 3, 32, 32, device=device)
        image2 = torch.randn(1, 3, 32, 32, device=device)
        
        loss_single = single_layer(image1, image2)
        loss_multi = multi_layer(image1, image2)
        
        # Multi-layer loss should generally be higher
        assert loss_multi >= loss_single
    
    def test_perceptual_loss_with_weights(self, feature_extractor, device):
        """Test perceptual loss with layer weights."""
        layers = ['layer1', 'layer2']
        weights = [1.0, 2.0]
        
        perceptual_loss = PerceptualLoss(feature_extractor, layers, weights)
        
        image1 = torch.randn(1, 3, 32, 32, device=device)
        image2 = torch.randn(1, 3, 32, 32, device=device)
        
        loss = perceptual_loss(image1, image2)
        
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestFeatureExtractor:
    """Test feature extractor implementation."""
    
    def test_feature_extractor_vgg_style(self, device):
        """Test VGG-style feature extractor."""
        # This would test against actual VGG if we have it implemented
        extractor = FeatureExtractor()
        
        image = torch.randn(2, 3, 224, 224, device=device)
        features = extractor(image)
        
        # Should return dictionary of features
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # All features should have correct batch size
        for layer_name, feature_map in features.items():
            assert feature_map.shape[0] == 2  # batch size


class TestStyleTransferConsistency:
    """Test consistency across style transfer components."""
    
    def test_all_losses_handle_batches(self, feature_extractor, device):
        """Test that all losses handle batch inputs correctly."""
        losses = [
            ContentLoss(),
            StyleLoss(),
            TotalVariationLoss(),
            PerceptualLoss(feature_extractor, ['layer1'])
        ]
        
        for loss_fn in losses:
            for batch_size in [1, 2, 4]:
                if isinstance(loss_fn, (ContentLoss, StyleLoss)):
                    # These take feature maps as input
                    input1 = torch.randn(batch_size, 64, 16, 16, device=device)
                    input2 = torch.randn(batch_size, 64, 16, 16, device=device)
                    result = loss_fn(input1, input2)
                elif isinstance(loss_fn, TotalVariationLoss):
                    # This takes images as input
                    input1 = torch.randn(batch_size, 3, 32, 32, device=device)
                    result = loss_fn(input1)
                else:  # PerceptualLoss
                    input1 = torch.randn(batch_size, 3, 32, 32, device=device)
                    input2 = torch.randn(batch_size, 3, 32, 32, device=device)
                    result = loss_fn(input1, input2)
                
                # All should return scalar
                assert result.dim() == 0
    
    def test_loss_gradients(self, feature_extractor, device):
        """Test that losses produce gradients for optimization."""
        # Test with images that require gradients
        image1 = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
        image2 = torch.randn(1, 3, 32, 32, device=device)
        
        losses = [
            TotalVariationLoss(),
            PerceptualLoss(feature_extractor, ['layer1'])
        ]
        
        for loss_fn in losses:
            if isinstance(loss_fn, TotalVariationLoss):
                loss = loss_fn(image1)
            else:
                loss = loss_fn(image1, image2)
            
            loss.backward()
            
            # Should have gradients
            assert image1.grad is not None
            assert torch.any(image1.grad != 0)
            
            # Reset gradients for next test
            image1.grad.zero_()
    
    def test_loss_ranges(self, feature_extractor, device):
        """Test that losses return reasonable ranges."""
        image1 = torch.randn(1, 3, 32, 32, device=device)
        image2 = torch.randn(1, 3, 32, 32, device=device)
        
        # TV loss should be reasonable for normalized images
        tv_loss = TotalVariationLoss()
        tv_result = tv_loss(torch.tanh(image1))  # Normalized to [-1, 1]
        assert 0 <= tv_result.item() <= 10  # Reasonable range
        
        # Perceptual loss should be positive
        perceptual_loss = PerceptualLoss(feature_extractor, ['layer1'])
        perceptual_result = perceptual_loss(image1, image2)
        assert perceptual_result.item() >= 0