"""
Integrated Gradients - Integra gradientes a lo largo de una ruta desde una línea base hasta la entrada.
"""

from typing import Literal
import torch
from torch import nn


class IntegratedGradients(nn.Module):
    """
    Computes Integrated Gradients for model interpretability.

    This class implements the Integrated Gradients method, which attributes the
    prediction of a model to its input features by integrating gradients along a
    straight path from a baseline to the input.

    More details can be found in: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """
        super().__init__()
        self.model = model

    def _initialize_baseline(
        self,
        channels: int,
        height: int,
        width: int,
        baseline: Literal["zero", "random"],
    ) -> torch.Tensor:
        """
        Initializes the baseline tensor.

        Args:
            channels: Number of channels of the image.
            height: Height of the image.
            width: Width of the image.
            baseline: Type of baseline to use.

        Returns:
            Baseline tensor of shape (channels, height, width).

        Raises:
            ValueError: If the value of 'baseline' is not valid.
        """
        match baseline:
            case "zero":
                return torch.zeros((channels, height, width))
            case "random":
                return torch.randn((channels, height, width))
            case _:
                raise ValueError("Please introduce a correct value for the baseline.")

    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        n_steps: int = 10,
        baseline: Literal["zero", "random"] = "zero",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            n_steps: Number of steps for the integration path from baseline to input.
            baseline: Type of baseline to use.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """
        batch_size, channels, height, width = x.shape
        
        # Initialize baseline tensor and expand to batch size
        baseline_tensor = self._initialize_baseline(channels, height, width, baseline)
        baseline_tensor = baseline_tensor.to(x.device)
        baseline_tensor = baseline_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Create scaled inputs along the path from baseline to input
        scaled_inputs = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            scaled_input = baseline_tensor + alpha * (x - baseline_tensor)
            scaled_inputs.append(scaled_input)
        
        # Stack all scaled inputs: (n_steps+1, batch, C, H, W)
        scaled_inputs = torch.stack(scaled_inputs)
        
        # Compute gradients for all scaled inputs
        gradients = []
        for i in range(n_steps + 1):
            scaled_input = scaled_inputs[i].clone().detach().requires_grad_(True)
            output = self.model(scaled_input)
            
            if target_class is None:
                target_class_tensor = output.argmax(dim=1)
            else:
                target_class_tensor = torch.tensor([target_class] * batch_size, device=x.device)
                
            score = output.gather(1, target_class_tensor.unsqueeze(1)).sum()
            score.backward()
            
            gradients.append(scaled_input.grad)
        
        # Average gradients (Riemann sum approximation)
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Multiply by (input - baseline)
        integrated_gradients = avg_gradients * (x - baseline_tensor)
        
        # Sum across channels to get attribution map
        attribution = integrated_gradients.abs().sum(dim=1)
        
        # Normalize
        attr_min = attribution.view(attribution.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        attr_max = attribution.view(attribution.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        attribution_norm = (attribution - attr_min) / (attr_max - attr_min + 1e-8)
        
        return attribution_norm