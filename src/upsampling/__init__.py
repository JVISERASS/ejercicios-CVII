"""
Upsampling methods for neural networks.

This module contains various upsampling techniques:
- Bed of nails upsampling
- Nearest neighbor interpolation
- Bilinear interpolation  
- Max unpooling
"""

from .bed_of_nails import bed_of_nails
from .nearest_neighbor import nearest_neighbor
from .bilinear_interpolation import bilinear_interpolation
from .max_unpooling import max_unpooling, MaxUnpool2d

__all__ = [
    'bed_of_nails',
    'nearest_neighbor',
    'bilinear_interpolation',
    'max_unpooling',
    'MaxUnpool2d',
]