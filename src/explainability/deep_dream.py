"""
DeepDream - Mejora de imagen estilo DeepDream maximizando activaciones.
"""

import torch
from torch import nn


class DeepDream:
    """DeepDream-style image enhancement by maximizing activations."""

    def __init__(self, model: nn.Module, feature_layer: nn.Module, steps: int = 20, lr: float = 0.01) -> None:
        self.model = model
        self.feature_layer = feature_layer
        self.steps = steps
        self.lr = lr
        self.activations = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            self.activations = output

        self.feature_layer.register_forward_hook(hook)

    def dream(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DeepDream enhancement to input image.
        
        Args:
            x: Input image tensor
            
        Returns:
            Enhanced image with emphasized features
        """
        x = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        
        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x)
            
            # Maximize the norm of activations (enhance patterns)
            loss = self.activations.norm()
            loss.backward()
            
            optimizer.step()
            
        return x.detach()