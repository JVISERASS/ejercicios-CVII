"""
SHAP - Aproximación simple de SHAP usando interpolaciones lineales aleatorias.
"""

import torch
from torch import nn


class SHAP(nn.Module):
    """Very small approximate SHAP-like estimator using random linear interpolations.

    Returns a normalized [batch, H, W] map.
    """

    def __init__(self, model: nn.Module, baseline: torch.Tensor | None = None, n_samples: int = 50) -> None:
        super().__init__()
        self.model = model
        self.baseline = baseline
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if self.baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self.baseline.to(x.device)
            if baseline.shape != x.shape:
                baseline = baseline.expand_as(x)
        
        attributions = torch.zeros_like(x)
        
        for _ in range(self.n_samples):
            # Random interpolation weight
            alpha = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
            
            # Interpolated input
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            if target_class is None:
                target_class_tensor = output.argmax(dim=1)
            else:
                target_class_tensor = torch.tensor([target_class] * x.size(0), device=x.device)
                
            score = output.gather(1, target_class_tensor.unsqueeze(1)).sum()
            score.backward()
            
            # Accumulate attribution
            attributions += interpolated.grad * (x - baseline)
        
        # Average over samples
        attributions = attributions / self.n_samples
        
        # Sum across channels and normalize
        attribution_map = attributions.abs().sum(dim=1)
        
        attr_min = attribution_map.view(attribution_map.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        attr_max = attribution_map.view(attribution_map.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        attribution_norm = (attribution_map - attr_min) / (attr_max - attr_min + 1e-8)
        
        return attribution_norm