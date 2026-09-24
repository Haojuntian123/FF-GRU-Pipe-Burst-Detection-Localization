# coding=utf-8
"""Baseline CNN model for detection and localization."""

from __future__ import annotations

import torch.nn as nn


class CNNClassifier(nn.Module):
    """Compact CNN for time-series classification."""

    def __init__(self, in_channel, hidden_channel1, hidden_channel2, out_channel, hidden_feature, out_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel1, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=hidden_channel1, out_channels=hidden_channel2, kernel_size=(5, 3), padding=2),
            nn.MaxPool2d(kernel_size=(5, 3)),
            nn.Conv2d(in_channels=hidden_channel2, out_channels=out_channel, kernel_size=(3, 3), padding=2),
        )
        self.adp = nn.AdaptiveMaxPool2d(output_size=1)
        self.classification = nn.Sequential(
            nn.Linear(in_features=out_channel, out_features=hidden_feature),
            nn.ReLU(),
            nn.Linear(in_features=hidden_feature, out_features=out_features),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        x = self.adp(x)
        return self.classification(x.flatten(1))


CNN = CNNClassifier


if __name__ == "__main__":
    model = CNNClassifier(in_channel=1, hidden_channel1=8, hidden_channel2=16, out_channel=32, hidden_feature=64, out_features=2)
    print(model)
