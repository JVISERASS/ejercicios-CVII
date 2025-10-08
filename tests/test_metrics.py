"""
Tests for evaluation metrics comparing with sklearn and PyTorch implementations.
"""

import pytest
import torch
import numpy as np
from tests.conftest import assert_tensors_close
from src.utils.metrics import (
    f1_score,
    precision_recall_f1,
    dice_score,
    iou_score,
    pixel_accuracy
)

try:
    from sklearn.metrics import f1_score as sklearn_f1
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestF1Score:
    """Test F1 score calculation."""
    
    def test_f1_perfect_predictions(self, device):
        """Test F1 score with perfect predictions."""
        preds = torch.tensor([0.9, 0.8, 0.1, 0.2], device=device)
        targets = torch.tensor([1, 1, 0, 0], device=device)
        
        result = f1_score(preds, targets, threshold=0.5)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_f1_worst_predictions(self, device):
        """Test F1 score with worst possible predictions."""
        preds = torch.tensor([0.1, 0.2, 0.9, 0.8], device=device)
        targets = torch.tensor([1, 1, 0, 0], device=device)
        
        result = f1_score(preds, targets, threshold=0.5)
        
        assert abs(result) < 1e-6  # Should be 0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_f1_vs_sklearn(self, device):
        """Compare our F1 score with sklearn implementation."""
        torch.manual_seed(42)
        preds = torch.rand(100, device=device)
        targets = torch.randint(0, 2, (100,), device=device)
        threshold = 0.5
        
        # Our implementation
        our_result = f1_score(preds, targets, threshold)
        
        # Sklearn implementation
        binary_preds = (preds > threshold).cpu().numpy().astype(int)
        sklearn_result = sklearn_f1(targets.cpu().numpy(), binary_preds)
        
        assert abs(our_result - sklearn_result) < 1e-6
    
    def test_f1_different_thresholds(self, device):
        """Test F1 score with different threshold values."""
        preds = torch.tensor([0.3, 0.7, 0.4, 0.8], device=device)
        targets = torch.tensor([0, 1, 0, 1], device=device)
        
        # With threshold 0.5: predictions [0, 1, 0, 1] -> perfect match
        result_05 = f1_score(preds, targets, threshold=0.5)
        
        # With threshold 0.6: predictions [0, 1, 0, 1] -> perfect match  
        result_06 = f1_score(preds, targets, threshold=0.6)
        
        assert abs(result_05 - 1.0) < 1e-6
        assert abs(result_06 - 1.0) < 1e-6


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 calculation."""
    
    def test_precision_recall_f1_perfect(self, device):
        """Test with perfect predictions."""
        preds = torch.tensor([0.9, 0.8, 0.1, 0.2], device=device)
        targets = torch.tensor([1, 1, 0, 0], device=device)
        
        result = precision_recall_f1(preds, targets, threshold=0.5)
        
        assert abs(result['precision'] - 1.0) < 1e-6
        assert abs(result['recall'] - 1.0) < 1e-6
        assert abs(result['f1'] - 1.0) < 1e-6
        assert abs(result['accuracy'] - 1.0) < 1e-6
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_precision_recall_f1_vs_sklearn(self, device):
        """Compare with sklearn implementations."""
        torch.manual_seed(42)
        preds = torch.rand(100, device=device)
        targets = torch.randint(0, 2, (100,), device=device)
        threshold = 0.5
        
        # Our implementation
        our_result = precision_recall_f1(preds, targets, threshold)
        
        # Sklearn implementation
        binary_preds = (preds > threshold).cpu().numpy().astype(int)
        targets_np = targets.cpu().numpy()
        
        sklearn_precision = precision_score(targets_np, binary_preds, zero_division=0)
        sklearn_recall = recall_score(targets_np, binary_preds, zero_division=0)
        sklearn_f1 = sklearn_f1(targets_np, binary_preds, zero_division=0)
        sklearn_accuracy = accuracy_score(targets_np, binary_preds)
        
        assert abs(our_result['precision'] - sklearn_precision) < 1e-6
        assert abs(our_result['recall'] - sklearn_recall) < 1e-6
        assert abs(our_result['f1'] - sklearn_f1) < 1e-6
        assert abs(our_result['accuracy'] - sklearn_accuracy) < 1e-6


class TestDiceScore:
    """Test Dice coefficient calculation."""
    
    def test_dice_perfect_overlap(self, device):
        """Test Dice coefficient with perfect overlap."""
        preds = torch.ones(2, 32, 32, device=device)
        targets = torch.ones(2, 32, 32, device=device)
        
        result = dice_score(preds, targets)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_dice_no_overlap(self, device):
        """Test Dice coefficient with no overlap."""
        preds = torch.zeros(2, 32, 32, device=device)
        targets = torch.ones(2, 32, 32, device=device)
        
        result = dice_score(preds, targets)
        
        # Should be close to 0 (smooth factor prevents exact 0)
        assert result < 0.01
    
    def test_dice_partial_overlap(self, device):
        """Test Dice coefficient with partial overlap."""
        preds = torch.zeros(1, 32, 32, device=device)
        targets = torch.zeros(1, 32, 32, device=device)
        
        # Set half of the pixels to 1 in both
        preds[0, :16, :] = 1
        targets[0, :16, :] = 1
        
        result = dice_score(preds, targets)
        
        # Should be close to 1 (perfect overlap in non-zero region)
        assert abs(result - 1.0) < 0.01
    
    def test_dice_vs_manual_calculation(self, device):
        """Test Dice coefficient against manual calculation."""
        preds = torch.zeros(1, 4, 4, device=device)
        targets = torch.zeros(1, 4, 4, device=device)
        
        # Set some specific pixels
        preds[0, :2, :2] = 1  # 4 pixels
        targets[0, 1:3, 1:3] = 1  # 4 pixels, 1 overlap
        
        result = dice_score(preds, targets)
        
        # Manual calculation: 2 * intersection / (sum1 + sum2)
        intersection = 1  # 1 overlapping pixel
        union = 4 + 4  # 4 + 4 pixels
        expected = (2 * intersection + 1e-8) / (union + 1e-8)
        
        assert abs(result - expected) < 1e-6


class TestIoUScore:
    """Test IoU score for segmentation."""
    
    def test_iou_perfect_overlap(self, device):
        """Test IoU with perfect overlap."""
        preds = torch.ones(2, 32, 32, device=device)
        targets = torch.ones(2, 32, 32, device=device)
        
        result = iou_score(preds, targets)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_iou_no_overlap(self, device):
        """Test IoU with no overlap."""
        preds = torch.zeros(2, 32, 32, device=device)
        targets = torch.ones(2, 32, 32, device=device)
        
        result = iou_score(preds, targets)
        
        # Should be close to 0 (smooth factor prevents exact 0)
        assert result < 0.01
    
    def test_iou_vs_manual_calculation(self, device):
        """Test IoU against manual calculation."""
        preds = torch.zeros(1, 4, 4, device=device)
        targets = torch.zeros(1, 4, 4, device=device)
        
        # Set some specific pixels
        preds[0, :2, :2] = 1  # 4 pixels
        targets[0, 1:3, 1:3] = 1  # 4 pixels, 1 overlap
        
        result = iou_score(preds, targets)
        
        # Manual calculation: intersection / union
        intersection = 1  # 1 overlapping pixel
        union = 4 + 4 - 1  # 4 + 4 - 1 overlap = 7
        expected = (intersection + 1e-8) / (union + 1e-8)
        
        assert abs(result - expected) < 1e-6


class TestPixelAccuracy:
    """Test pixel accuracy calculation."""
    
    def test_pixel_accuracy_perfect(self, device):
        """Test pixel accuracy with perfect predictions."""
        preds = torch.tensor([[0, 1, 2], [1, 0, 2]], device=device)
        targets = torch.tensor([[0, 1, 2], [1, 0, 2]], device=device)
        
        result = pixel_accuracy(preds, targets)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_pixel_accuracy_worst(self, device):
        """Test pixel accuracy with worst predictions."""
        preds = torch.tensor([[1, 0, 1], [0, 1, 0]], device=device)
        targets = torch.tensor([[0, 1, 0], [1, 0, 1]], device=device)
        
        result = pixel_accuracy(preds, targets)
        
        assert abs(result) < 1e-6  # Should be 0
    
    def test_pixel_accuracy_partial(self, device):
        """Test pixel accuracy with partial correctness."""
        preds = torch.tensor([[0, 1, 2], [1, 0, 1]], device=device)
        targets = torch.tensor([[0, 1, 2], [1, 0, 2]], device=device)
        
        result = pixel_accuracy(preds, targets)
        
        # 5 out of 6 pixels correct
        expected = 5.0 / 6.0
        assert abs(result - expected) < 1e-6
    
    def test_pixel_accuracy_different_shapes(self, device):
        """Test pixel accuracy with different tensor shapes."""
        # Test with 3D tensors (batch dimension)
        preds = torch.randint(0, 3, (4, 32, 32), device=device)
        targets = preds.clone()  # Perfect match
        
        result = pixel_accuracy(preds, targets)
        
        assert abs(result - 1.0) < 1e-6