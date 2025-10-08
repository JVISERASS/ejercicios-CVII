"""
Gradient Ascent - Ascenso de gradiente en la entrada para maximizar la puntuación de una clase objetivo.
"""

import torch
from torch import nn


class GradienteAscendente:
    """Gradient ascent on the input to maximize the score of a target class.

    This class is intended for visualization (produces images), not saliency maps.
    """

    def __init__(self, model: nn.Module, steps: int = 20, lr: float = 0.1) -> None:
        self.model = model
        self.steps = steps
        self.lr = lr

    def generate(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Generate input that maximizes the target class score.
        
        Args:
            x: Initial input tensor
            target_class: Target class to maximize
            
        Returns:
            Optimized input tensor
        """
        x = x.clone().detach().requires_grad_(True)
        
        for _ in range(self.steps):
            output = self.model(x)
            loss = output[0, target_class]
            loss.backward()
            
            # Gradient ascent step
            x.data += self.lr * x.grad.data
            x.grad.zero_()
            
        return x.detach()