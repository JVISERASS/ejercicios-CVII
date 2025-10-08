"""
DeconvNet - Atribución estilo DeconvNet que restringe gradientes hacia atrás a positivos.
"""

import torch
from torch import nn


class DeconvNet(nn.Module):
    """DeconvNet style attribution (clamps backward gradient to positives)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.hooks = []
        
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def _register_hooks(self):
        def hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_backward_hook(hook))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
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