"""Test topology variants without hydraulic observations."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class FeatureFusionTest(unittest.TestCase):
    def test_dense_topology_replaces_both_relation_matrices(self):
        import torch

        from fusion_models.feature_fusion import FeatureFusion

        pressure_adj = torch.tensor([[0., 1.], [0., 0.]])
        flow_graph = torch.tensor([[1., 0.], [0., 1.]])
        fusion = FeatureFusion(flow_graph, pressure_adj, 2, 3, 2, "dense_topology")
        self.assertTrue(torch.allclose(fusion.lp.adj, torch.full((2, 2), 0.5)))
        self.assertTrue(torch.allclose(fusion.laq.g, torch.full((2, 2), 0.5)))
        output = fusion(torch.ones(2, 3, 2), torch.ones(2, 2))
        self.assertEqual(tuple(output.shape), (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
