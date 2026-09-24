"""Optional joint fine-tuning step for pretrained task-specific models.

The main cross-model comparison uses independent training. For the alternative
strategy, pretrain the two models separately, then pass both models and their
parameters to this step with an optimizer spanning both parameter sets.
"""

from __future__ import annotations

import torch


def joint_loss(detection_logits, localization_logits, detection_target, localization_target, detection_weight=1.0, localization_weight=1.0):
    """Compute the weighted joint objective from two logits tensors."""
    detection = torch.nn.functional.cross_entropy(detection_logits, detection_target)
    localization = torch.nn.functional.cross_entropy(localization_logits, localization_target)
    return detection_weight * detection + localization_weight * localization


def train_joint_step(detection_model, localization_model, detection_batch,
                     localization_batch, optimizer, device,
                     detection_weight=1.0, localization_weight=1.0):
    """Update both task models with equally weighted losses by default.

    Each batch is an (inputs, targets) pair. Localization inputs contain burst
    samples; no hard detection decision enters either training computation graph.
    """
    detection_model.train()
    localization_model.train()
    detection_inputs, detection_target = detection_batch
    localization_inputs, localization_target = localization_batch
    move = lambda data: tuple(x.to(device) for x in data) if isinstance(data, (tuple, list)) else data.to(device)
    optimizer.zero_grad(set_to_none=True)
    detection_logits = detection_model(move(detection_inputs))
    localization_logits = localization_model(move(localization_inputs))
    loss = joint_loss(
        detection_logits,
        localization_logits,
        detection_target.to(device),
        localization_target.to(device),
        detection_weight=detection_weight,
        localization_weight=localization_weight,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
