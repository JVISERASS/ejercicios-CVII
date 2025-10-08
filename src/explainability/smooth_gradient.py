"""
SmoothGrad - Promedia gradientes sobre muestras con ruido para reducir artifacts.
"""

import torch
from torch import nn


class SmoothGradient(nn.Module):
    """SmoothGrad: average gradients over noisy samples."""

    def __init__(self, model: nn.Module, n_samples: int = 50, noise_level: float = 0.1) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        accumulated = torch.zeros_like(x, device=x.device)

        for _ in range(self.n_samples):
            # Add noise to input
            noise = torch.randn_like(x) * self.noise_level
            x_noisy = x + noise
            x_noisy = x_noisy.clone().detach().requires_grad_(True)
            
            output = self.model(x_noisy)
            
            if target_class is None:
                target_class_tensor = output.argmax(dim=1)
            else:
                target_class_tensor = torch.tensor([target_class] * x.size(0), device=x.device)
                
            score = output.gather(1, target_class_tensor.unsqueeze(1)).sum()
            score.backward()
            
            accumulated += x_noisy.grad
        
        avg = accumulated / float(self.n_samples)
        saliency = avg.abs().sum(dim=1)

        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        
        return saliency_norm