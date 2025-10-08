"""
Metrics - Métricas de evaluación para computer vision.
"""

import torch
from typing import Dict, Any


def f1_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute F1 score for binary classification.
    
    Args:
        preds: Prediction probabilities [N]
        targets: Ground truth labels [N] (0 or 1)
        threshold: Threshold for converting probabilities to predictions
        
    Returns:
        F1 score
    """
    binary_preds = (preds > threshold).float()
    
    tp = ((binary_preds == 1) & (targets == 1)).sum().float()
    fp = ((binary_preds == 1) & (targets == 0)).sum().float()
    fn = ((binary_preds == 0) & (targets == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return float(f1)


def precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        preds: Prediction probabilities [N]
        targets: Ground truth labels [N] (0 or 1)
        threshold: Threshold for converting probabilities to predictions
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    binary_preds = (preds > threshold).float()
    
    tp = ((binary_preds == 1) & (targets == 1)).sum().float()
    fp = ((binary_preds == 1) & (targets == 0)).sum().float()
    fn = ((binary_preds == 0) & (targets == 1)).sum().float()
    tn = ((binary_preds == 0) & (targets == 0)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8) -> float:
    """
    Compute Dice coefficient for segmentation tasks.
    
    Args:
        preds: Predicted segmentation masks [N, H, W] or [N, C, H, W]
        targets: Ground truth masks [N, H, W] or [N, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return float(dice)


def iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8) -> float:
    """
    Compute IoU (Jaccard index) for segmentation tasks.
    
    Args:
        preds: Predicted segmentation masks [N, H, W] or [N, C, H, W]
        targets: Ground truth masks [N, H, W] or [N, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)

def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    """
    Computes Chamfer distance between two point clouds.
    Args:
        pc1: [N1, 3]
        pc2: [N2, 3]
    Returns:
        Chamfer distance
    """
    dist1 = torch.cdist(pc1, pc2).min(dim=1)[0]
    dist2 = torch.cdist(pc2, pc1).min(dim=1)[0]
    return dist1.mean().item() + dist2.mean().item()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute pixel accuracy for segmentation tasks.
    
    Args:
        preds: Predicted class labels [N, H, W]
        targets: Ground truth class labels [N, H, W]
        
    Returns:
        Pixel accuracy
    """
    correct = (preds == targets).sum().float()
    total = targets.numel()
    
    return float(correct / total)