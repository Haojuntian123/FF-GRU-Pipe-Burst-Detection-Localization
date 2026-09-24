from __future__ import annotations

import torch.nn as nn

from models.GCN import GCNClassifier

from .feature_fusion import FeatureFusion


class FFGCNClassifier(nn.Module):
    """Feature-fusion GCN classifier."""

    def __init__(
        self,
        flow_graph,
        pressure_adj,
        sensors,
        gcn_hidden,
        linear_hidden,
        laq_hidden,
        out_features,
        adj,
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
        sequence_length: int = 60,
        fusion_mode: str = "cascade",
    ):
        super().__init__()
        self.fusion = FeatureFusion(flow_graph, pressure_adj, sensors, sequence_length, laq_hidden, fusion_mode)
        self.backbone = GCNClassifier(
            in_features=sensors,
            gcn_hidden=gcn_hidden,
            linear_hidden=linear_hidden,
            out_features=out_features,
            dropout=dropout,
            node_readout=node_readout,
            time_readout=time_readout,
            readout_order=readout_order,
            classifier_mode=classifier_mode,
            feature_norm=feature_norm,
            use_residual=use_residual,
            temporal_conv_kernel=temporal_conv_kernel,
            temporal_conv_layers=temporal_conv_layers,
            temporal_hidden=temporal_hidden,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_readout=temporal_readout,
            temporal_input=temporal_input,
            adj=adj,
        )

    def forward(self, data):
        x, q = data
        x = self.fusion(x, q)
        return self.backbone(x)


FFGCN = FFGCNClassifier
