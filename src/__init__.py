"""
Computer Vision Exercises - Organized Implementation

This package contains organized implementations of various computer vision algorithms and techniques:

- explainability: Neural network interpretability methods
- representations: 3D data representations and geometric processing  
- upsampling: Image upsampling techniques
- models: Classic CNN architectures
- style_transfer: Neural style transfer loss functions
- utils: General utility functions and metrics

Each module is self-contained and focuses on a specific area of computer vision.
"""

from . import explainability
from . import representations
from . import upsampling
from . import models
from . import style_transfer
from . import utils

__version__ = "1.0.0"
__author__ = "Computer Vision Course"

__all__ = [
    'explainability',
    'representations', 
    'upsampling',
    'models',
    'style_transfer',
    'utils',
]