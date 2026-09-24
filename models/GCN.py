# coding=utf-8
"""Baseline GCN model for detection and localization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_axis(x, dim, mode):
    if mode == "max":
        return torch.max(x, dim=dim).values
    if mode == "mean":
        return torch.mean(x, dim=dim)
    if mode == "last":
        index = x.shape[dim] - 1
        return x.select(dim, index)
    if mode == "meanmax":
        mean = torch.mean(x, dim=dim)
        max_value = torch.max(x, dim=dim).values
        return torch.cat([mean, max_value], dim=-1)
    raise ValueError(f"unsupported readout mode: {mode}")


def _readout_multiplier(mode):
    return 2 if mode == "meanmax" else 1


def build_normalized_adjacency(adj, add_self_loops: bool = True):
    """Return the symmetric normalized adjacency matrix."""

    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {tuple(adj.shape)}")
    device = adj.device
    size = adj.shape[0]
    a = adj.clone()
    if add_self_loops:
        a = a + torch.eye(size, device=device, dtype=a.dtype)
    degree = a.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    scale = degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0)
    return (a * scale).to(device=device, dtype=torch.float32)


class GraphConvLayer(nn.Module):
    """A small graph convolution layer."""

    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj_norm):
        xw = self.lin(x)
        return torch.einsum("ij,bjf->bif", adj_norm, xw)


class TemporalConvBlock(nn.Module):
    """A gated temporal convolution block."""

    def __init__(self, channels, kernel_size, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, padding=padding)
        self.dropout = dropout

    def forward(self, x):
        residual = x
        gated = self.conv(x)
        value, gate = torch.chunk(gated, 2, dim=1)
        x = value * torch.sigmoid(gate)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual


class GCNClassifier(nn.Module):
    """GCN-based classifier with configurable readout."""

    def __init__(
        self,
        in_features,
        gcn_hidden,
        linear_hidden,
        out_features,
        dropout: float = 0.0,
        node_readout: str = "mean",
        time_readout: str = "mean",
        readout_order: str = "node_time",
        classifier_mode: str = "pooled",
        feature_norm: str = "none",
        use_residual: bool = False,
        temporal_conv_kernel: int = 0,
        temporal_conv_layers: int = 0,
        temporal_hidden: int = 0,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        temporal_readout: str = "last",
        temporal_input: str = "flatten",
        adj=None,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.sensors_num = in_features
        self.dropout = dropout
        self.node_readout = node_readout.lower()
        self.time_readout = time_readout.lower()
        self.readout_order = readout_order.lower()
        self.classifier_mode = classifier_mode.lower()
        self.feature_norm = feature_norm.lower()
        self.use_residual = bool(use_residual)
        self.temporal_conv_kernel = int(temporal_conv_kernel or 0)
        self.temporal_conv_layers = int(temporal_conv_layers or 0)
        self.temporal_hidden = int(temporal_hidden or 0)
        self.temporal_layers = int(temporal_layers or 1)
        self.temporal_dropout = temporal_dropout
        self.temporal_readout = temporal_readout.lower()
        self.temporal_input = temporal_input.lower()
        if self.node_readout not in {"mean", "max", "meanmax"}:
            raise ValueError(f"unsupported node_readout: {node_readout}")
        if self.time_readout not in {"mean", "max", "last", "meanmax"}:
            raise ValueError(f"unsupported time_readout: {time_readout}")
        if self.readout_order not in {"node_time", "time_node", "dual"}:
            raise ValueError(f"unsupported readout_order: {readout_order}")
        if self.classifier_mode not in {"pooled", "flatten_nodes"}:
            raise ValueError(f"unsupported classifier_mode: {classifier_mode}")
        if self.feature_norm not in {"none", "layernorm"}:
            raise ValueError(f"unsupported feature_norm: {feature_norm}")
        if self.temporal_conv_kernel not in {0, 1} and self.temporal_conv_kernel % 2 == 0:
            raise ValueError("temporal_conv_kernel must be odd")

        self.gcn1 = GraphConvLayer(1, gcn_hidden)
        self.gcn2 = GraphConvLayer(gcn_hidden, gcn_hidden)
        norm_cls = nn.LayerNorm if self.feature_norm == "layernorm" else None
        self.norm1 = norm_cls(gcn_hidden) if norm_cls else nn.Identity()
        self.norm2 = norm_cls(gcn_hidden) if norm_cls else nn.Identity()

        self.temporal_conv = nn.ModuleList()
        if self.temporal_conv_layers > 0 and self.temporal_conv_kernel > 0:
            for _ in range(self.temporal_conv_layers):
                self.temporal_conv.append(TemporalConvBlock(gcn_hidden, self.temporal_conv_kernel, dropout=self.dropout))

        base_readout_dim = gcn_hidden * _readout_multiplier(self.node_readout) * _readout_multiplier(self.time_readout)
        if self.classifier_mode == "flatten_nodes":
            classifier_in = in_features * gcn_hidden * _readout_multiplier(self.time_readout)
        else:
            classifier_in = base_readout_dim * (2 if self.readout_order == "dual" else 1)
        self.temporal = None
        if self.temporal_hidden > 0:
            if self.temporal_input not in {"flatten", "mean", "max"}:
                raise ValueError(f"unsupported temporal_input: {temporal_input}")
            if self.temporal_readout not in {"last", "mean", "max"}:
                raise ValueError(f"unsupported temporal_readout: {temporal_readout}")
            temporal_in_features = gcn_hidden * in_features if self.temporal_input == "flatten" else gcn_hidden
            self.temporal = nn.GRU(
                input_size=temporal_in_features,
                hidden_size=self.temporal_hidden,
                num_layers=self.temporal_layers,
                batch_first=True,
                dropout=temporal_dropout if self.temporal_layers > 1 else 0.0,
            )
            classifier_in = self.temporal_hidden

        self.classification = nn.Sequential(
            nn.Linear(classifier_in, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_features),
        )
        self.register_buffer("adj_norm", None)
        if adj is not None:
            self.set_graph(adj, add_self_loops=add_self_loops)

    def set_graph(self, adj, add_self_loops: bool = True):
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, dtype=torch.float32)
        adj = adj.to(next(self.parameters()).device, dtype=torch.float32)
        if adj.shape != (self.sensors_num, self.sensors_num):
            raise ValueError(f"adj must be {(self.sensors_num, self.sensors_num)}, got {tuple(adj.shape)}")
        self.adj_norm = build_normalized_adjacency(adj, add_self_loops=add_self_loops)

    def encode(self, x):
        if self.adj_norm is None:
            raise RuntimeError("adj_norm is not set")
        if x.dim() != 3:
            raise ValueError(f"GCNClassifier expects (B, T, S), got {tuple(x.shape)}")
        batch_size, seq_len, sensors = x.shape
        if sensors != self.sensors_num:
            raise ValueError(f"GCNClassifier expects {self.sensors_num} sensors, got {sensors}")

        h = x.reshape(batch_size * seq_len, sensors, 1)
        h = self.gcn1(h, self.adj_norm)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h2 = self.gcn2(h, self.adj_norm)
        if self.use_residual:
            h2 = h2 + h
        h = self.norm2(h2)
        h = F.relu(h)
        h = h.reshape(batch_size, seq_len, sensors, -1)
        if self.temporal_conv:
            hc = h.permute(0, 2, 3, 1).reshape(batch_size * sensors, -1, seq_len)
            for block in self.temporal_conv:
                hc = block(hc)
            h = hc.reshape(batch_size, sensors, -1, seq_len).permute(0, 3, 1, 2)
        return h

    def classify_encoded(self, h):
        batch_size, seq_len, sensors, _ = h.shape

        if self.temporal is not None:
            if self.temporal_input == "flatten":
                temporal_input = h.reshape(batch_size, seq_len, -1)
            elif self.temporal_input == "max":
                temporal_input = torch.max(h, dim=2).values
            else:
                temporal_input = torch.mean(h, dim=2)
            temporal_output, _ = self.temporal(temporal_input)
            if self.temporal_readout == "max":
                g = torch.max(temporal_output, dim=1).values
            elif self.temporal_readout == "mean":
                g = torch.mean(temporal_output, dim=1)
            else:
                g = temporal_output[:, -1, :]
            return self.classification(g)

        if self.classifier_mode == "flatten_nodes":
            node_tokens = _reduce_axis(h, dim=1, mode=self.time_readout)
            g = node_tokens.reshape(batch_size, -1)
            return self.classification(g)

        spatial_tokens = _reduce_axis(h, dim=2, mode=self.node_readout)
        spatial_graph = _reduce_axis(spatial_tokens, dim=1, mode=self.time_readout)
        temporal_tokens = _reduce_axis(h, dim=1, mode=self.time_readout)
        temporal_graph = _reduce_axis(temporal_tokens, dim=1, mode=self.node_readout)
        if self.readout_order == "time_node":
            g = temporal_graph
        elif self.readout_order == "dual":
            g = torch.cat([spatial_graph, temporal_graph], dim=1)
        else:
            g = spatial_graph
        return self.classification(g)

    def forward(self, x):
        h = self.encode(x)
        return self.classify_encoded(h)


GCN = GCNClassifier
