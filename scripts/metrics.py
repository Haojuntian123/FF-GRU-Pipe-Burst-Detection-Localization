"""Metrics for burst detection and localization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def detection_metrics(y_true, y_pred, positive_label: int = 1) -> dict:
    """Return the binary detection metrics.

    DAC is overall detection accuracy, REC is burst recall, FAR is the
    false-alarm rate among normal windows, and F1 is the positive-class F1.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    negative = 0 if positive_label != 0 else 1
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true == negative) & (y_pred == positive_label)))
    tn = int(np.sum((y_true == negative) & (y_pred != positive_label)))
    return {
        "DAC": float(accuracy_score(y_true, y_pred)),
        "PRE": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "REC": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "FAR": float(fp / (fp + tn)) if fp + tn else 0.0,
        "F1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def localization_metrics(y_true, y_pred) -> dict:
    """Return partition-level localization accuracy and macro-F1."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "LAC": float(accuracy_score(y_true, y_pred)),
        "macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def summarize_seeds(rows) -> dict:
    """Summarize numeric metric dictionaries as mean and sample SD over seeds."""
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary


def format_mean_sd(rows) -> dict:
    """Format numeric seed summaries as mean +/- SD strings."""
    summary = summarize_seeds(rows)
    keys = sorted({key[:-5] for key in summary if key.endswith("_mean")})
    return {key: f"{summary[key + '_mean']:.4f} +/- {summary[key + '_sd']:.4f} (n={len(rows)})" for key in keys}


def diagnostic_metrics(
    detection_sample_id,
    detection_true,
    detection_pred,
    localization_sample_id,
    localization_true,
    localization_pred,
) -> dict:
    """Compute detection, localization, and sample-matched CAC together."""
    result = {}
    result.update(detection_metrics(detection_true, detection_pred))
    result.update(localization_metrics(localization_true, localization_pred))
    result["CAC"] = coordinated_accuracy(
        detection_sample_id,
        detection_true,
        detection_pred,
        localization_sample_id,
        localization_true,
        localization_pred,
    )
    result["CAC_PB"] = coordinated_accuracy(
        detection_sample_id,
        detection_true,
        detection_pred,
        localization_sample_id,
        localization_true,
        localization_pred,
        burst_only=True,
    )
    return result


def write_json(payload: dict, output_path: str | Path) -> None:
    """Write a JSON metric report, creating its parent directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def coordinated_accuracy(
    detection_sample_id,
    detection_true,
    detection_pred,
    localization_sample_id,
    localization_true,
    localization_pred,
    burst_only: bool = False,
) -> float:
    """Return full-chain accuracy over all windows or true burst windows."""
    det_id = np.asarray(detection_sample_id)
    loc_id = np.asarray(localization_sample_id)
    det_true = np.asarray(detection_true)
    det_pred = np.asarray(detection_pred)
    loc_true = np.asarray(localization_true)
    loc_pred = np.asarray(localization_pred)
    if len({a.shape[0] for a in (det_id, det_true, det_pred)}) != 1:
        raise ValueError("Detection arrays must have equal length")
    if len({a.shape[0] for a in (loc_id, loc_true, loc_pred)}) != 1:
        raise ValueError("Localization arrays must have equal length")
    if len(np.unique(det_id)) != det_id.size or len(np.unique(loc_id)) != loc_id.size:
        raise ValueError("sample_id must identify each evaluation window exactly once")
    loc_keys = set(loc_id.tolist())
    if not np.isin(det_true, (0, 1)).all():
        raise ValueError("Detection ground truth must use 0=normal and 1=burst")
    burst_keys = {key for key, label in zip(det_id.tolist(), det_true.tolist()) if label == 1}
    if burst_keys != loc_keys:
        missing_from_loc = burst_keys - loc_keys
        missing_from_det = loc_keys - burst_keys
        raise ValueError(
            "Localization must cover exactly the true burst sample_id set "
            f"(missing_from_localization={len(missing_from_loc)}, "
            f"unexpected_localization={len(missing_from_det)})"
        )
    loc_index = {key: i for i, key in enumerate(loc_id.tolist())}
    correct = (det_true == 0) & (det_pred == 0)
    for i, key in enumerate(det_id.tolist()):
        if det_true[i] == 1:
            j = loc_index[key]
            correct[i] = det_pred[i] == 1 and loc_true[j] == loc_pred[j]
    evaluated = correct[det_true == 1] if burst_only else correct
    if not evaluated.size:
        raise ValueError("No samples in the requested CAC evaluation subset")
    return float(evaluated.mean())
