from __future__ import annotations

import torch.nn as nn

from .feature_fusion import FeatureFusion


class SELayer(nn.Module):
    """Squeeze-and-excitation block."""

    def __init__(self, features, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(features, features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(features // reduction, features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)


class Block(nn.Module):
    """Dense feature extraction block."""

    def __init__(self, in_features, hidden1, hidden2, hidden3, out_features, dropout=0.5):
        super().__init__()
        self.short = nn.Sequential(nn.BatchNorm1d(in_features), nn.ReLU())
        self.density = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.density(self.short(x))


class FFFADenseNetClassifier(nn.Module):
    """Feature-fusion FA-DenseNet classifier."""

    def __init__(self, g, pressure_adj, sensors, in_features, out_features, serious_len, fusion_mode="cascade", block_dropout=0.5):
        super().__init__()
        self.fusion = FeatureFusion(g, pressure_adj, sensors, serious_len, 2 * sensors, fusion_mode)
        self.input = nn.Linear(in_features * serious_len, 2 * serious_len)
        self.block1 = Block(2 * serious_len, 4 * serious_len, 12 * serious_len, 4 * serious_len, 2 * serious_len, dropout=block_dropout)
        self.se1 = SELayer(2 * serious_len, serious_len)
        self.block2 = Block(2 * serious_len, 6 * serious_len, 16 * serious_len, 6 * serious_len, 2 * serious_len, dropout=block_dropout)
        self.se2 = SELayer(2 * serious_len, serious_len)
        self.block3 = Block(2 * serious_len, 9 * serious_len, 14 * serious_len, 9 * serious_len, 2 * serious_len, dropout=block_dropout)
        self.se3 = SELayer(2 * serious_len, serious_len)
        self.block4 = Block(2 * serious_len, 6 * serious_len, 16 * serious_len, 6 * serious_len, 2 * serious_len, dropout=block_dropout)
        self.se4 = SELayer(2 * serious_len, serious_len)
        self.block5 = Block(2 * serious_len, 18 * serious_len, 10 * serious_len, 2 * serious_len, out_features, dropout=block_dropout)
        self.output = nn.Identity()

    def forward(self, data):
        x, q = data
        x = self.fusion(x, q)
        x = x.reshape((x.shape[0], -1))
        x = self.input(x)
        d1 = x
        x = self.block1(d1)
        x = self.se1(x)
        d2 = x + d1
        x = self.block2(d2)
        x = self.se2(x)
        d3 = x + d1 + d2
        x = self.block3(d3)
        x = self.se3(x)
        d4 = x + d1 + d2 + d3
        x = self.block4(d4)
        x = self.se4(x)
        d5 = x + d1 + d2 + d3 + d4
        x = self.block5(d5)
        return self.output(x)


FFFADenseNet = FFFADenseNetClassifier
FFFA_DenseNet = FFFADenseNetClassifier
