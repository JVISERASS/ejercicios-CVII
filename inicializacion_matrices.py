import os
from typing import Literal

import numpy as np
from skimage import segmentation, color
from skimage.future import graph
from skimage import io
from skimage.feature import selective_search 
ss = selective_search

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# Bed of Nails
def bed_of_nails(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Bed of nails upsampling (nearest neighbor with zeros).
    Args:
        x: Tensor [N, C, H, W]
        stride: Upsampling factor
    Returns:
        Upsampled tensor
    """
    N, C, H, W = x.shape
    output = torch.zeros(N, C, H*stride, W*stride, device=x.device, dtype = x.dtype)
    output[:,:,::stride,::stride] = x
    return output

# Nearest Neighbor Interpolation
def nearest_neighbor(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Nearest neighbor interpolation.
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out)
    Returns:
        Upsampled tensor
    """
    N, C, H, W = x.shape
    h_out, w_out = size
    idx_y = torch.floor(torch.linspace(0, H-1, h_out, device=x.device)).long()
    idx_x = torch.floor(torch.linspace(0, W-1, w_out, device=x.device)).long()
    output = torch.zeros(N, C, h_out, w_out)
    for i in range(h_out):
        for j in range(w_out):
            output[:, :, i, j] = x[:, :, idx_y[i], idx_x[j]]
    return output

# Bilinear Interpolation
def bilinear_interpolation(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Bilinear interpolation.
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out)
    Returns:
        Upsampled tensor
    """
    N, C, H, W = x.shape
    H_out, W_out = size
    grid_y = torch.linspace(0, H - 1, H_out, device=x.device)
    grid_x = torch.linspace(0, W - 1, W_out, device=x.device)
    y0 = torch.floor(grid_y).long()
    x0 = torch.floor(grid_x).long()
    y1 = torch.clamp(y0 + 1, max=H - 1)
    x1 = torch.clamp(x0 + 1, max=W - 1)
    wy = (grid_y - y0.float()).view(-1, 1)
    wx = (grid_x - x0.float()).view(1, -1)

    v00 = x[:, :, y0.unsqueeze(1), x0]
    v01 = x[:, :, y0.unsqueeze(1), x1]
    v10 = x[:, :, y1.unsqueeze(1), x0]
    v11 = x[:, :, y1.unsqueeze(1), x1]

    out = (
        (1 - wy) * (1 - wx) * v00 +
        (1 - wy) * wx * v01 +
        wy * (1 - wx) * v10 +
        wy * wx * v11
    )
    return out

# Max Unpooling
def max_unpooling(x: torch.Tensor, indices: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Max unpooling operation.
    Args:
        x: Pooled tensor
        indices: Indices from max pooling
        kernel_size: Pooling kernel size
        stride: Pooling stride
    Returns:
        Unpooled tensor
    """
    # unpool = nn.MaxUnpool2d(kernel_size, stride)
    # Manual implementation of Max Unpooling
    # x: [N, C, H, W], indices: [N, C, H, W]

    N, C, H, W = x.shape

    h_out = H * stride
    w_out = W * stride

    output = torch.zeros(N, C, h_out, w_out, device=x.device, dtype = x.dtype)

    x_flat = x.view(N, C, -1)
    idxs_flat = indices.view(N, C, -1)
    output_flat = output.view(N, C, -1)

    for n in range(N):
        for c in range(C):
            output_flat[n,c].scatter_(0, idxs_flat[n,c], x_flat[n,c])
    
    output = output_flat.view(N, C, h_out, w_out)
    return output