from __future__ import annotations

import torch.nn as nn

from .feature_fusion import FeatureFusion


class FFCNNClassifier(nn.Module):
    """Feature-fusion CNN classifier."""

    def __init__(self, g, pressure_adj, sensors, in_channel, hidden_channel1, hidden_channel2, out_channel, hidden_feature, out_features, sequence_length=60, fusion_mode="cascade"):
        super().__init__()
        self.fusion = FeatureFusion(g, pressure_adj, sensors, sequence_length, 3 * sensors, fusion_mode)
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

    def forward(self, data):
        x, q = data
        x = self.fusion(x, q)
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        x = self.adp(x)
        return self.classification(x.flatten(1))


FFCNN = FFCNNClassifier
