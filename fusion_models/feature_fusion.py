# coding=utf-8
"""Feature-fusion modules for the improved models."""

from __future__ import annotations

import torch
import torch.nn as nn


class LP(nn.Module):
    """Pressure-topology gate used before flow guidance in the cascade."""

    def __init__(self, features, sequence, sensors, adj):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(features * sequence, sensors), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, dtype=torch.float32)
        adj = adj.to(dtype=torch.float32)
        # The pressure graph keeps self-links and uses row normalization so
        # nodes with different numbers of reachable monitors remain comparable.
        adj = torch.maximum(adj, torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device))
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj", adj)
        self.f = nn.Linear(sensors, features)

    def forward(self, x):
        nx = x.reshape((x.shape[0], -1))
        out = nx.mm(self.weight)
        out = torch.sigmoid(self.f(out.mm(self.adj)))
        return out.unsqueeze(dim=1) * x


class FeatureFusion(nn.Module):
    """Selectable pressure/flow fusion for backbone ablations."""

    MODES = {"cascade", "pressure_only", "flow_only", "concat", "dense_topology"}

    def __init__(self, g, pressure_adj, sensors, sequence_length, laq_hidden, mode="cascade"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported fusion mode: {mode}")
        if mode == "dense_topology":
            pressure_adj = torch.ones_like(torch.as_tensor(pressure_adj))
            g = torch.ones_like(torch.as_tensor(g))
        self.mode = mode
        self.lp = LP(sensors, sequence_length, sensors, pressure_adj)
        self.laq = LAQ(g, sensors, laq_hidden)
        self.projection = nn.Linear(2 * sensors, sensors) if mode == "concat" else None

    def forward(self, x, q):
        if self.mode == "pressure_only":
            return self.lp(x)
        if self.mode == "flow_only":
            return self.laq(x, q)
        if self.mode == "concat":
            return self.projection(torch.cat((self.lp(x), self.laq(x, q)), dim=-1))
        return self.laq(self.lp(x), q)


class AQ(nn.Module):
    """Flow-guided attention gate."""

    def __init__(self, qsensors, hidden, output):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(qsensors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
            nn.Sigmoid(),
        )

    def forward(self, x, q):
        w = self.f(q)
        return x * w.unsqueeze(dim=1)


class LAQ(nn.Module):
    """Flow-guided sensor weighting with graph projection."""

    def __init__(self, g, sensors, hidden, mode: str = "mul", strength: float = 1.0, guide_norm: str = "none"):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(sensors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, sensors),
            nn.Sigmoid(),
        )
        if not torch.is_tensor(g):
            g = torch.tensor(g, dtype=torch.float32)
        g = g.to(dtype=torch.float32)
        # Flow-to-pressure relations are bipartite; normalize each flow row.
        g = g / g.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("g", g)
        self.mode = str(mode).lower()
        self.strength = float(strength)
        self.guide_norm = str(guide_norm).lower()
        self.norm = nn.LayerNorm(sensors) if self.guide_norm == "layernorm" else nn.Identity()
        if self.mode not in {"mul", "residual"}:
            raise ValueError(f"unsupported LAQ mode: {mode}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"LAQ strength must be in [0, 1], got {strength}")
        if self.guide_norm not in {"none", "layernorm"}:
            raise ValueError(f"unsupported LAQ guide_norm: {guide_norm}")

    def sensor_weights(self, q):
        if q.dim() not in {2, 3}:
            raise ValueError(f"LAQ expects q with shape [B, Q] or [B, T, Q], got {tuple(q.shape)}")
        w = torch.matmul(q, self.g)
        w = self.norm(w)
        return self.f(w)

    def sensor_scale(self, q):
        w = self.sensor_weights(q)
        if self.mode == "residual":
            return 1.0 + self.strength * (w - 0.5)
        return (1.0 - self.strength) + self.strength * w

    def forward(self, x, q):
        scale = self.sensor_scale(q)
        if scale.dim() == 2:
            return x * scale.unsqueeze(dim=1)
        if scale.dim() == 3:
            if x.dim() != 3:
                raise ValueError(f"sequence LAQ expects x with shape [B, T, S], got {tuple(x.shape)}")
            if x.shape != scale.shape:
                raise ValueError(f"sequence LAQ shape mismatch: x={tuple(x.shape)} scale={tuple(scale.shape)}")
            return x * scale
        raise ValueError(f"unsupported LAQ scale shape: {tuple(scale.shape)}")
