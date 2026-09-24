from __future__ import annotations

import torch.nn as nn

from .feature_fusion import FeatureFusion


class FFGRUClassifier(nn.Module):
    """Feature-fusion GRU classifier."""

    def __init__(self, g, pressure_adj, in_features, gru_hidden, linear_hidden, laq_hidden, out_features, sequence_length=60, num_layers=3, dropout=0.0, fusion_mode="cascade"):
        super().__init__()
        self.fusion = FeatureFusion(g, pressure_adj, in_features, sequence_length, laq_hidden, fusion_mode)
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classification = nn.Sequential(
            nn.Linear(gru_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_features),
        )

    def forward(self, data):
        x, q = data
        x = self.fusion(x, q)
        gru_x, _ = self.gru(x)
        last_step = gru_x[:, -1, :]
        return self.classification(last_step)


FFGRU = FFGRUClassifier
