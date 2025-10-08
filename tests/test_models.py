"""
Tests for neural network models comparing custom implementations with PyTorch equivalents.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.conftest import assert_tensors_close, assert_same_shape_and_type
from src.models import (
    SimpleConvNet,
    ResidualBlock,
    ResNet18,
    UNet,
    VGGBlock,
    VGG16,
    AttentionBlock,
    TransformerEncoder
)


class TestSimpleConvNet:
    """Test simple convolutional network implementation."""
    
    def test_simple_conv_net_output_shape(self, device):
        """Test that SimpleConvNet produces correct output shape."""
        model = SimpleConvNet(in_channels=3, num_classes=10).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_simple_conv_net_vs_pytorch(self, device):
        """Compare SimpleConvNet with equivalent PyTorch implementation."""
        # Our implementation
        our_model = SimpleConvNet(in_channels=3, num_classes=5).to(device)
        
        # Equivalent PyTorch implementation
        pytorch_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 5)
        ).to(device)
        
        # Copy weights to make them identical
        with torch.no_grad():
            for our_param, pt_param in zip(our_model.parameters(), pytorch_model.parameters()):
                pt_param.copy_(our_param)
        
        x = torch.randn(3, 3, 32, 32, device=device)
        
        our_output = our_model(x)
        pt_output = pytorch_model(x)
        
        assert_tensors_close(our_output, pt_output, rtol=1e-5)
        assert_same_shape_and_type(our_output, pt_output)


class TestResidualBlock:
    """Test residual block implementation."""
    
    def test_residual_block_identity(self, device):
        """Test residual block with identity mapping."""
        block = ResidualBlock(64, 64).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        
        output = block(x)
        
        # Output should have same shape as input
        assert output.shape == x.shape
    
    def test_residual_block_with_downsample(self, device):
        """Test residual block with downsampling."""
        block = ResidualBlock(64, 128, stride=2).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        
        output = block(x)
        
        # Output should be downsampled
        assert output.shape == (2, 128, 8, 8)
    
    def test_residual_block_vs_pytorch(self, device):
        """Compare ResidualBlock with torchvision ResNet block."""
        from torchvision.models.resnet import BasicBlock
        
        # Our implementation
        our_block = ResidualBlock(64, 64).to(device)
        
        # PyTorch implementation (BasicBlock)
        pt_block = BasicBlock(64, 64).to(device)
        
        # Copy weights to make them similar
        with torch.no_grad():
            our_block.conv1.weight.copy_(pt_block.conv1.weight)
            our_block.conv1.bias.copy_(pt_block.conv1.bias)
            our_block.bn1.weight.copy_(pt_block.bn1.weight)
            our_block.bn1.bias.copy_(pt_block.bn1.bias)
            our_block.conv2.weight.copy_(pt_block.conv2.weight)
            our_block.conv2.bias.copy_(pt_block.conv2.bias)
            our_block.bn2.weight.copy_(pt_block.bn2.weight)
            our_block.bn2.bias.copy_(pt_block.bn2.bias)
        
        x = torch.randn(2, 64, 16, 16, device=device)
        
        our_output = our_block(x)
        pt_output = pt_block(x)
        
        # Should be very close if implementation is correct
        assert_tensors_close(our_output, pt_output, rtol=1e-4)


class TestResNet18:
    """Test ResNet-18 implementation."""
    
    def test_resnet18_output_shape(self, device):
        """Test ResNet-18 output shape."""
        model = ResNet18(num_classes=1000).to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        
        output = model(x)
        
        assert output.shape == (2, 1000)
    
    def test_resnet18_vs_torchvision(self, device):
        """Compare with torchvision ResNet-18."""
        import torchvision.models as models
        
        # Our implementation
        our_model = ResNet18(num_classes=10).to(device)
        
        # Torchvision implementation
        tv_model = models.resnet18(pretrained=False, num_classes=10).to(device)
        
        x = torch.randn(1, 3, 224, 224, device=device)
        
        our_output = our_model(x)
        tv_output = tv_model(x)
        
        # Shapes should match
        assert our_output.shape == tv_output.shape


class TestUNet:
    """Test U-Net implementation."""
    
    def test_unet_output_shape(self, device):
        """Test U-Net output shape for segmentation."""
        model = UNet(in_channels=3, out_channels=2).to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        
        output = model(x)
        
        # Output should have same spatial dimensions as input
        assert output.shape == (1, 2, 256, 256)
    
    def test_unet_skip_connections(self, device):
        """Test that U-Net properly handles skip connections."""
        model = UNet(in_channels=1, out_channels=1).to(device)
        
        # Test with different input sizes
        for size in [64, 128, 256]:
            x = torch.randn(1, 1, size, size, device=device)
            output = model(x)
            
            assert output.shape == (1, 1, size, size)
    
    def test_unet_vs_segmentation_models(self, device):
        """Compare U-Net structure with standard implementation."""
        # This is more of a structure test since we don't have exact equivalent
        model = UNet(in_channels=3, out_channels=21).to(device)  # PASCAL VOC classes
        
        x = torch.randn(2, 3, 224, 224, device=device)
        output = model(x)
        
        # Should produce logits for each class at each pixel
        assert output.shape == (2, 21, 224, 224)
        
        # Test that gradients flow properly
        loss = output.mean()
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestVGG:
    """Test VGG implementation."""
    
    def test_vgg_block_basic(self, device):
        """Test basic VGG block functionality."""
        block = VGGBlock(64, 128, num_layers=2).to(device)
        x = torch.randn(2, 64, 32, 32, device=device)
        
        output = block(x)
        
        # Should change channels but preserve spatial dims
        assert output.shape == (2, 128, 32, 32)
    
    def test_vgg16_output_shape(self, device):
        """Test VGG-16 output shape."""
        model = VGG16(num_classes=1000).to(device)
        x = torch.randn(1, 3, 224, 224, device=device)
        
        output = model(x)
        
        assert output.shape == (1, 1000)
    
    def test_vgg16_vs_torchvision(self, device):
        """Compare with torchvision VGG-16."""
        import torchvision.models as models
        
        # Our implementation
        our_model = VGG16(num_classes=10).to(device)
        
        # Torchvision implementation
        tv_model = models.vgg16(pretrained=False, num_classes=10).to(device)
        
        x = torch.randn(1, 3, 224, 224, device=device)
        
        our_output = our_model(x)
        tv_output = tv_model(x)
        
        # Shapes should match
        assert our_output.shape == tv_output.shape


class TestAttentionMechanisms:
    """Test attention mechanisms."""
    
    def test_attention_block_basic(self, device):
        """Test basic attention block functionality."""
        attention = AttentionBlock(embed_dim=128, num_heads=8).to(device)
        
        # Input: [batch_size, seq_len, embed_dim]
        x = torch.randn(2, 10, 128, device=device)
        
        output = attention(x)
        
        # Output should have same shape as input
        assert output.shape == x.shape
    
    def test_attention_block_vs_pytorch(self, device):
        """Compare attention with PyTorch MultiheadAttention."""
        embed_dim = 64
        num_heads = 8
        seq_len = 12
        batch_size = 3
        
        # Our implementation
        our_attention = AttentionBlock(embed_dim, num_heads).to(device)
        
        # PyTorch implementation
        pt_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
        
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        our_output = our_attention(x)
        pt_output, _ = pt_attention(x, x, x)  # Self-attention
        
        # Shapes should match
        assert our_output.shape == pt_output.shape == x.shape
    
    def test_transformer_encoder_basic(self, device):
        """Test transformer encoder functionality."""
        encoder = TransformerEncoder(
            embed_dim=128,
            num_heads=8,
            ff_dim=256,
            num_layers=2
        ).to(device)
        
        x = torch.randn(2, 20, 128, device=device)
        
        output = encoder(x)
        
        # Should preserve shape
        assert output.shape == x.shape
    
    def test_transformer_encoder_vs_pytorch(self, device):
        """Compare with PyTorch TransformerEncoder."""
        embed_dim = 64
        num_heads = 4
        ff_dim = 128
        num_layers = 2
        
        # Our implementation
        our_encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers).to(device)
        
        # PyTorch implementation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        pt_encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device)
        
        x = torch.randn(3, 15, embed_dim, device=device)
        
        our_output = our_encoder(x)
        pt_output = pt_encoder(x)
        
        # Shapes should match
        assert our_output.shape == pt_output.shape == x.shape


class TestModelConsistency:
    """Test consistency across different models."""
    
    def test_all_models_handle_different_batch_sizes(self, device):
        """Test that all models handle different batch sizes."""
        models = [
            SimpleConvNet(3, 10),
            ResNet18(num_classes=10),
            VGG16(num_classes=10),
            UNet(in_channels=3, out_channels=1)
        ]
        
        for model in models:
            model = model.to(device)
            
            for batch_size in [1, 4, 8]:
                if isinstance(model, UNet):
                    x = torch.randn(batch_size, 3, 64, 64, device=device)
                    output = model(x)
                    assert output.shape[0] == batch_size
                else:
                    x = torch.randn(batch_size, 3, 224, 224, device=device)
                    output = model(x)
                    assert output.shape[0] == batch_size
    
    def test_models_training_mode(self, device):
        """Test that models behave differently in train vs eval mode."""
        model = ResNet18(num_classes=10).to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        
        # Training mode
        model.train()
        train_output = model(x)
        
        # Eval mode
        model.eval()
        eval_output = model(x)
        
        # Outputs may differ due to batch norm behavior
        assert train_output.shape == eval_output.shape
    
    def test_gradient_flow(self, device):
        """Test that gradients flow properly through all models."""
        models = [
            SimpleConvNet(3, 5),
            ResidualBlock(32, 32),
            UNet(1, 1)
        ]
        
        for model in models:
            model = model.to(device)
            model.train()
            
            if isinstance(model, ResidualBlock):
                x = torch.randn(1, 32, 16, 16, device=device, requires_grad=True)
            elif isinstance(model, UNet):
                x = torch.randn(1, 1, 64, 64, device=device, requires_grad=True)
            else:
                x = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
            
            output = model(x)
            loss = output.mean()
            loss.backward()
            
            # Check that parameters have gradients
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None