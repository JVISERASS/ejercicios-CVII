import torch

import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """
    Content loss for style transfer.
    Measures the MSE between the target and input feature maps.
    """
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        return F.mse_loss(input, self.target)

class StyleLoss(nn.Module):
    """
    Style loss for style transfer.
    Measures the MSE between the Gram matrices of the target and input feature maps.
    """
    def __init__(self, target_feature):
        super().__init__()
        self.target_gram = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        return F.mse_loss(G, self.target_gram)


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss for style transfer.
    Encourages spatial smoothness in the generated image.
    """
    def forward(self, input):
        return torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))