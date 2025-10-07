import torch
import torch.nn.functional as F


class GradienteAscendente:
    """Gradient ascent on the input to maximize the score of a target class.

    This class is intended for visualization (produces images), not saliency maps.
    """

    def __init__(self, model, steps: int = 20, lr: float = 0.1) -> None:
        self.model = model
        self.steps = steps
        self.lr = lr

    def generate(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            output = self.model(x)
            loss = output[0, target_class]
            loss.backward()
            x.data += self.lr * x.grad.data
            x.grad.zero_()
        return x.detach()

class InversionDeCaracteristicas:
    """Feature inversion: optimize an input to match target features on a given layer."""

    def __init__(self, model, feature_layer, steps: int = 100, lr: float = 0.1) -> None:
        self.model = model
        self.feature_layer = feature_layer
        self.steps = steps
        self.lr = lr
        self.activations = None
        self._register_hook()

    def _register_hook(self) -> None:
        def hook(module, input, output):
            self.activations = output

        self.feature_layer.register_forward_hook(hook)

    def invert(self, target_features: torch.Tensor, x_init: torch.Tensor) -> torch.Tensor:
        x = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        for _ in range(self.steps):
            output = self.model(x)
            loss = F.mse_loss(self.activations, target_features)
            loss.backward()
            optimizer.step()
        return x.detach()


class DeepDream:
    """DeepDream-style image enhancement by maximizing activations."""

    def __init__(self, model, feature_layer, steps: int = 20, lr: float = 0.01) -> None:
        self.model = model
        self.feature_layer = feature_layer
        self.steps = steps
        self.lr = lr
        self.activations = None
        self._register_hook()

    def _register_hook(self) -> None:
        def hook(module, input, output):
            self.activations = output

        self.feature_layer.register_forward_hook(hook)

    def dream(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        for _ in range(self.steps):
            output = self.model(x)
            loss = self.activations.norm()
            loss.backward()
            optimizer.step()
        return x.detach()