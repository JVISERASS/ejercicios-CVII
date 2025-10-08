"""
Feature Inversion - Inversión de características: optimiza una entrada para coincidir con características objetivo en una capa dada.
"""

import torch
import torch.nn.functional as F
from torch import nn


class InversionDeCaracteristicas:
    """Feature inversion: optimize an input to match target features on a given layer."""

    def __init__(self, model: nn.Module, feature_layer: nn.Module, steps: int = 100, lr: float = 0.1) -> None:
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

    def invert(self, target_features: torch.Tensor, x_init: torch.Tensor) -> torch.Tensor:
        """
        Invert features to match target activations.
        
        Args:
            target_features: Target feature activations to match
            x_init: Initial input guess
            
        Returns:
            Optimized input that produces similar features
        """
        x = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        
        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x)
            
            # Compute loss between current and target features
            loss = F.mse_loss(self.activations, target_features)
            loss.backward()
            
            optimizer.step()
            
        return x.detach()