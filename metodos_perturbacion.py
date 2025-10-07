import torch
import torch.nn.functional as F
from torch import nn

from utils import normalize_tensor


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
        pass

class Occlusion(nn.Module):
    """Occlusion sensitivity: systematically occlude patches and measure score drop.

    Returns normalized [batch, H, W]."""

    def __init__(self, model: nn.Module, patch_size: int = 8) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_class: int | None = None, stride: int | None = None) -> torch.Tensor:
        pass


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
        pass