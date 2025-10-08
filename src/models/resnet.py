"""
ResNet - Redes Residuales con skip connections.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Bloque residual básico para ResNet-18/34."""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bloque bottleneck para ResNet-50/101/152."""
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Red ResNet con bloques residuales."""
    
    def __init__(self, block, num_blocks: list, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_planes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Crea una capa con múltiples bloques residuales."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Funciones de construcción para diferentes variantes
def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes: int = 1000) -> ResNet:
    """ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)