"""
Occlusion - Sensibilidad por oclusión: ocluye parches sistemáticamente y mide la caída de puntuación.
"""

import torch
from torch import nn


class Occlusion(nn.Module):
    """Occlusion sensitivity: systematically occlude patches and measure score drop.

    Returns normalized [batch, H, W]."""

    def __init__(self, model: nn.Module, patch_size: int = 8) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_class: int | None = None, stride: int | None = None) -> torch.Tensor:
        if stride is None:
            stride = self.patch_size // 2
            
        batch_size, channels, height, width = x.shape
        
        # Get original predictions
        original_output = self.model(x)
        if target_class is None:
            target_class_tensor = original_output.argmax(dim=1)
            original_scores = original_output.gather(1, target_class_tensor.unsqueeze(1)).squeeze()
        else:
            target_class_tensor = torch.tensor([target_class] * batch_size, device=x.device)
            original_scores = original_output.gather(1, target_class_tensor.unsqueeze(1)).squeeze()
        
        # Initialize importance map
        importance_map = torch.zeros(batch_size, height, width, device=x.device)
        
        # Slide occlusion window
        for y in range(0, height - self.patch_size + 1, stride):
            for x_pos in range(0, width - self.patch_size + 1, stride):
                # Create occluded input
                occluded_input = x.clone()
                occluded_input[:, :, y:y+self.patch_size, x_pos:x_pos+self.patch_size] = 0
                
                # Get occluded predictions
                occluded_output = self.model(occluded_input)
                occluded_scores = occluded_output.gather(1, target_class_tensor.unsqueeze(1)).squeeze()
                
                # Calculate importance (score drop)
                score_diff = original_scores - occluded_scores
                
                # Assign importance to the occluded region
                importance_map[:, y:y+self.patch_size, x_pos:x_pos+self.patch_size] = torch.maximum(
                    importance_map[:, y:y+self.patch_size, x_pos:x_pos+self.patch_size],
                    score_diff.unsqueeze(-1).unsqueeze(-1)
                )
        
        # Normalize
        imp_min = importance_map.view(batch_size, -1).min(dim=1)[0].view(-1, 1, 1)
        imp_max = importance_map.view(batch_size, -1).max(dim=1)[0].view(-1, 1, 1)
        importance_norm = (importance_map - imp_min) / (imp_max - imp_min + 1e-8)
        
        return importance_norm