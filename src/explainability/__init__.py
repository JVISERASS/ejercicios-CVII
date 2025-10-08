"""
Explainability methods for neural network interpretation.

This module contains various methods for explaining neural network decisions:
- Gradient-based methods (Saliency, Guided Backprop, etc.)
- Perturbation-based methods (SHAP, Occlusion, RISE)
- Input optimization methods (Gradient Ascent, Feature Inversion, DeepDream)
"""

# Gradient-based methods
from .saliency_maps import saliency_map, SaliencyMaps, SaliencyMapsMax, SaliencyMapsL2
from .input_x_gradient import InputXGradient
from .guided_backprop import GuidedBackpropagation
from .deconv_net import DeconvNet
from .smooth_gradient import SmoothGradient
from .integrated_gradients import IntegratedGradients

# Perturbation-based methods
from .shap import SHAP
from .occlusion import Occlusion
from .rise import RISE

# Input optimization methods
from .gradient_ascent import GradienteAscendente
from .feature_inversion import InversionDeCaracteristicas
from .deep_dream import DeepDream

__all__ = [
    # Gradient-based
    'saliency_map',
    'SaliencyMaps',
    'SaliencyMapsMax',
    'SaliencyMapsL2',
    'InputXGradient', 
    'GuidedBackpropagation',
    'DeconvNet',
    'SmoothGradient',
    'IntegratedGradients',
    
    # Perturbation-based
    'SHAP',
    'Occlusion',
    'RISE',
    
    # Input optimization
    'GradienteAscendente',
    'InversionDeCaracteristicas',
    'DeepDream',
]