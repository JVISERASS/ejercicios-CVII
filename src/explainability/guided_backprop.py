"""
Guided Backpropagation - Modifica el pase hacia atrás de ReLU para propagar solo gradientes positivos.
"""

import torch
from torch import nn


class GuidedBackpropagation(nn.Module):
    """Guided Backpropagation: modify ReLU backward pass so only positive gradients
    are propagated. Returns normalized [batch, H, W].
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.activation_maps: list[torch.Tensor] = []
        self.hooks = []
        
        # Ensure ReLUs are not inplace
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            if isinstance(module, nn.ReLU):
                self.activation_maps.append(out)

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                # Only pass positive gradients
                if len(self.activation_maps) > 0:
                    activation = self.activation_maps.pop()
                    # Clamp negative gradients and activations
                    return (torch.clamp(grad_out[0], min=0.0) * (activation > 0).float(),)
                else:
                    return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activation_maps.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        if not any(isinstance(m, nn.ReLU) for m in self.model.modules()):
            raise ValueError("Model must contain ReLU layers for Guided Backpropagation")

        self._register_hooks()
        try:
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
        finally:
            self._remove_hooks()