"""
Normalization Utils - Utilidades para normalización de tensores.
"""

import torch


def normalize_tensor(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the explanation tensor to [0, 1] range.

    Args:
        explanation: Explanation tensor with dimensions [batch, height, width].

    Returns:
        Normalized explanation with the same dimensions in range [0, 1].
    """
    max_ = torch.amax(explanation, dim=(1, 2), keepdim=True)
    min_ = torch.amin(explanation, dim=(1, 2), keepdim=True)

    return (explanation - min_) / (max_ - min_ + 1e-8)


def standardize_tensor(tensor: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None) -> torch.Tensor:
    """
    Standardize tensor to zero mean and unit variance.
    
    Args:
        tensor: Input tensor
        mean: Optional mean values, computed if None
        std: Optional std values, computed if None
        
    Returns:
        Standardized tensor
    """
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    
    return (tensor - mean) / (std + 1e-8)


def min_max_normalize(tensor: torch.Tensor, dim: tuple = None) -> torch.Tensor:
    """
    Min-max normalization to [0, 1] range.
    
    Args:
        tensor: Input tensor
        dim: Dimensions over which to compute min/max
        
    Returns:
        Normalized tensor in [0, 1] range
    """
    if dim is None:
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)
    
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def z_score_normalize(tensor: torch.Tensor, dim: tuple = None) -> torch.Tensor:
    """
    Z-score normalization (zero mean, unit variance).
    
    Args:
        tensor: Input tensor
        dim: Dimensions over which to compute mean/std
        
    Returns:
        Z-score normalized tensor
    """
    if dim is None:
        mean = tensor.mean()
        std = tensor.std()
    else:
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True)
    
    return (tensor - mean) / (std + 1e-8)