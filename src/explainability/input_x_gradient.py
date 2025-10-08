"""
Input × Gradient - Método de atribución que multiplica entrada por gradiente.
"""

import torch
from torch import nn


class InputXGradient(nn.Module):
    """Input * Gradient attribution. Returns normalized [batch, H, W]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D [batch, channels, height, width]")
        
        x = x.clone().detach().requires_grad_(True)
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1)
        else:
            target_class = torch.tensor([target_class] * x.size(0), device=x.device)
            
        score = output.gather(1, target_class.unsqueeze(1)).sum()
        score.backward()
        
        saliency = (x * x.grad).abs().sum(dim=1)
        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        
        return saliency_norm