"""
RISE - Randomized Input Sampling for Explanation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RISE(nn.Module):
    """Randomized Input Sampling for Explanation (RISE).

    Produces a normalized [batch, H, W] saliency map.
    """

    def __init__(self, model: nn.Module, n_masks: int = 1000, mask_size: int = 7, p: float = 0.5) -> None:
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.mask_size = mask_size
        self.p = p

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Initialize importance maps
        importance_maps = torch.zeros(batch_size, height, width, device=x.device)
        
        # Generate random masks
        for _ in range(self.n_masks):
            # Create random mask
            mask = torch.rand(height + self.mask_size, width + self.mask_size, device=x.device) < self.p
            
            # Random crop to original size
            start_h = torch.randint(0, self.mask_size + 1, (1,)).item()
            start_w = torch.randint(0, self.mask_size + 1, (1,)).item()
            mask = mask[start_h:start_h + height, start_w:start_w + width]
            
            # Apply mask to input
            masked_input = x * mask.unsqueeze(0).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(masked_input)
                
                if target_class is None:
                    target_class_tensor = output.argmax(dim=1)
                else:
                    target_class_tensor = torch.tensor([target_class] * batch_size, device=x.device)
                    
                scores = output.gather(1, target_class_tensor.unsqueeze(1)).squeeze()
            
            # Accumulate weighted importance
            for b in range(batch_size):
                importance_maps[b] += scores[b] * mask.float()
        
        # Average over masks
        importance_maps = importance_maps / self.n_masks
        
        # Normalize
        imp_min = importance_maps.view(batch_size, -1).min(dim=1)[0].view(-1, 1, 1)
        imp_max = importance_maps.view(batch_size, -1).max(dim=1)[0].view(-1, 1, 1)
        importance_norm = (importance_maps - imp_min) / (imp_max - imp_min + 1e-8)
        
        return importance_norm