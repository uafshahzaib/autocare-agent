"""
A small convolutional network for dashboard warning-light classification,
architecturally in the LeCun et al. (1998) LeNet lineage (conv -> pool ->
conv -> pool -> fully-connected) — the simplest architecture that is still
a genuine, trainable CNN rather than a linear classifier on raw pixels.
A production system at BMW's scale would swap this for a modern pretrained
backbone (e.g. a fine-tuned ResNet or ViT), but the interface below
(forward pass, save/load) would be unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vision.dataset import CLASSES


class WarningLightCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw logits
