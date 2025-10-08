"""
Saliency Maps - Mapas de relevancia basados en gradientes vanilla y sus variantes.
Incluye la implementación básica y variaciones del método.
"""

import torch
from torch import nn


def saliency_map(model: nn.Module, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
    """
    Computes gradients of the model output with respect to input. As we want to return a
    saliency map, we must obtain the maximum of the absolute values along the channels
    dimension.

    Args:
        model: Model we want to explain.
        x: Input tensor. Dimensions: [batch, channels, height, width].
        target_class: Class for which backward pass is computed. If None, then we assume
            it is the one with maximum score.

    Returns:
        Saliency map of the model output with respect to input. 
        Dimensions: [batch, height, width] if batch > 1, or [height, width] if batch = 1.

    Raises:
        RuntimeError: If the gradients calculated with the backward are None.
    """
    x.requires_grad_(True)
    model.zero_grad()
    
    output = model(x) # dim [batch, num_classes]
    
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Crear un tensor one-hot para la clase objetivo
    target = torch.zeros_like(output)
    target[torch.arange(output.size(0)), target_class] = 1
    
    # Calcular gradientes
    output.backward(gradient=target)
    
    # Obtener el máximo del valor absoluto a lo largo de los canales
    saliency = torch.abs(x.grad).max(dim=1)[0]
    
    # If batch size is 1, squeeze the batch dimension
    if x.shape[0] == 1:
        saliency = saliency.squeeze(0)
    
    # Limpiar gradientes
    x.grad = None
    x.requires_grad_(False)
    
    return saliency

class SaliencyMaps(nn.Module):
    """Simple vanilla gradient saliency maps with normalization.

    Returns a normalized [batch, H, W] map.
    """

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
        
        saliency = x.grad.abs().sum(dim=1)
        saliency_min = saliency.view(saliency.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        
        return saliency_norm


class SaliencyMapsMax(nn.Module):
    """Saliency maps using max across channels instead of sum.
    
    This is the variant from utils.py that uses .max(dim=1)[0] instead of .sum(dim=1).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D [batch, channels, height, width]")
        
        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1)
            
        scores = output[torch.arange(x.shape[0]), target_class].sum()
        scores.backward()
        
        # Usar max en lugar de sum (como en utils.py)
        saliency = x.grad.abs().max(dim=1)[0] #dim [batch, height, width] el [0] es porque .max() devuelve (valores, indices)
        
        if saliency is None:
            raise RuntimeError("Gradients are None")
        
        return saliency


class SaliencyMapsL2(nn.Module):
    """Saliency maps using L2 norm across channels."""

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
        
        # Usar L2 norm en lugar de L1
        saliency = x.grad.norm(dim=1)
        
        # Normalizar
        saliency_min = torch.amin(saliency, dim=(1, 2)).view(-1, 1, 1) # es lo mismo que .view(-1,1,1).min()[0].view(-1,1,1)
        saliency_max = torch.amax(saliency, dim=(1, 2)).view(-1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        
        return saliency_norm