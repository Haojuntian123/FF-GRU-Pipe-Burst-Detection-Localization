"""Test the two-model training step on synthetic tensors."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class JointTrainingTest(unittest.TestCase):
    def test_step_updates_both_task_models(self):
        import torch

        from scripts.joint_training import train_joint_step

        detection = torch.nn.Linear(3, 2)
        localization = torch.nn.Linear(3, 4)
        optimizer = torch.optim.SGD(
            list(detection.parameters()) + list(localization.parameters()), lr=0.1
        )
        before = [parameter.detach().clone() for model in (detection, localization)
                  for parameter in model.parameters()]
        loss = train_joint_step(
            detection, localization,
            (torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 1])),
            (torch.randn(4, 3), torch.tensor([0, 1, 2, 3])),
            optimizer, torch.device("cpu"),
        )
        self.assertGreater(loss, 0)
        after = [parameter for model in (detection, localization)
                 for parameter in model.parameters()]
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
