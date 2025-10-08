"""
Max Unpooling - Operación de max unpooling para revertir max pooling.
"""

import torch
from torch import nn


def max_unpooling(x: torch.Tensor, indices: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Max unpooling operation.
    
    Args:
        x: Pooled tensor [N, C, H, W]
        indices: Indices from max pooling operation [N, C, H, W]
        kernel_size: Pooling kernel size used in original max pooling
        stride: Pooling stride used in original max pooling
        
    Returns:
        Unpooled tensor with original spatial dimensions
    """
    N, C, H, W = x.shape

    h_out = H * stride
    w_out = W * stride

    output = torch.zeros(N, C, h_out, w_out, device=x.device, dtype=x.dtype)

    # Flatten for easier indexing
    x_flat = x.view(N, C, -1)
    idxs_flat = indices.view(N, C, -1)
    output_flat = output.view(N, C, -1)

    # Use scatter_ to place values at the correct indices
    output_flat.scatter_(2, idxs_flat, x_flat)
    output = output_flat.view(N, C, h_out, w_out)
    return output


class MaxUnpool2d(nn.Module):
    """
    Max Unpooling layer that can be used as a PyTorch module.
    """
    
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return max_unpooling(x, indices, self.kernel_size, self.stride)
    
    
def max_unpooling(x: torch.Tensor, indices: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Max unpooling operation.
    
    Args:
        x: Pooled tensor [N, C, H, W]
        indices: Indices from max pooling operation [N, C, H, W]
        kernel_size: Pooling kernel size used in original max pooling
        stride: Pooling stride used in original max pooling
        
    Returns:
        Unpooled tensor with original spatial dimensions
    """
    N, C, H, W = x.shape
    h_out = H * stride
    w_out = W*stride
    output = torch.zeros(N,C,h_out,w_out)
    idx_flat = indices.view(N,C,-1)
    x_falt = x.view(N,C,-1)
    output_flat = output.view(N,C,-1)
    
    output_flat.scatter_(2, idx_flat, x_falt)
    output = output_flat.view(N,C,h_out, w_out)