# coding=utf-8
"""Baseline GRU model for detection and localization."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """GRU-based sequence classifier."""

    def __init__(self, in_features, gru_hidden, linear_hidden, out_features, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
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

    def forward(self, x):
        gru_x, _ = self.gru(x)
        last_step = gru_x[:, -1, :]
        return self.classification(last_step)


GRU = GRUClassifier


if __name__ == "__main__":
    model = GRUClassifier(in_features=6, gru_hidden=32, linear_hidden=64, out_features=2)
    print(model)
