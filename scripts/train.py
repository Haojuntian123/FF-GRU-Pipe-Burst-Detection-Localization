"""Training and evaluation entrypoint."""

from __future__ import annotations

import csv
import copy
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import wntr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.dataset import CreateGraph, MyAlarmData, MyAlarmDataQ, MyLocationData, MyLocationDataQ
from fusion_models.FFGCN import FFGCNClassifier
from fusion_models.FFCNN import FFCNNClassifier
from fusion_models.FFFADenseNet import FFFADenseNetClassifier
from fusion_models.FFGRU import FFGRUClassifier
from models.CNN import CNNClassifier
from models.FADenseNet import FADenseNet
from models.GCN import GCNClassifier
from models.GRU import GRUClassifier
from scripts.metrics import detection_metrics, localization_metrics

SUPPORTED_TASKS = {"detection", "localization"}
MODEL_NAME_ALIASES = {
    "GRU": "GRU",
    "CNN": "CNN",
    "GCN": "GCN",
    "FA-DENSENET": "FA-DENSENET",
    "FADENSENET": "FA-DENSENET",
    "FF-GRU": "FF-GRU",
    "FF-CNN": "FF-CNN",
    "FF-GCN": "FF-GCN",
    "FF-FA-DENSENET": "FF-FA-DENSENET",
    "FFFADENSENET": "FF-FA-DENSENET",
    "FFFA-DENSENET": "FF-FA-DENSENET",
    "FFFA_DENSENET": "FF-FA-DENSENET",
}
SUPPORTED_MODELS = tuple(sorted(set(MODEL_NAME_ALIASES.values())))


def _require_prepartitioned_data():
    if bool(_get_value("DATA_SPLIT", default=False)):
        raise ValueError(
            "DATA_SPLIT is unsupported: assign normal source periods and burst parent events "
            "to train/validation/test before windowing, then configure separate files"
        )


def _device() -> torch.device:
    device = getattr(config, "DEVICE", torch.device("cpu"))
    return device if isinstance(device, torch.device) else torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_value(*names, default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().upper().replace("_", "-")
    normalized = MODEL_NAME_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}. Supported values: {supported}")
    return normalized


def _resolve_sensor_ids(wn, raw_values: Sequence):
    resolved = []
    for value in raw_values or []:
        if isinstance(value, int):
            resolved.append(wn.node_name_list[value])
        else:
            resolved.append(str(value))
    return resolved


def _resolve_pressure_feature_count() -> int:
    value = _get_value("NUM_PRESSURE_SENSORS", "INPUT_FEATURES", "IN_FEATURES", "N_FEATURES", default=0)
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _resolve_num_classes(task: str) -> int:
    if task == "detection":
        return 2
    return int(_get_value("NUM_CLASSES", "N_CLASSES", "OUTPUT_CLASSES", default=0) or 0)


def _build_graphs(device: torch.device):
    network_file = _get_value("NETWORK_FILE", "MODEL_FILE", "INP_FILE")
    if not network_file:
        raise ValueError("NETWORK_FILE or MODEL_FILE must be set in config.py")
    wn = wntr.network.WaterNetworkModel(str(network_file))
    graph_builder = CreateGraph(wn)
    pressure_sensor_ids = _resolve_sensor_ids(wn, _get_value("PRESSURE_SENSOR_INDEXES", "PRESSURE_SENSORS", "PRESSURE_MONITORS", default=[]))
    flow_sensor_ids = _resolve_sensor_ids(wn, _get_value("FLOW_SENSOR_INDEXES", "FLOW_SENSORS", default=[]))
    if not pressure_sensor_ids:
        pressure_sensor_ids = _resolve_sensor_ids(wn, _get_value("MONITORS", default=[]))
    if not pressure_sensor_ids:
        pressure_sensor_ids = wn.junction_name_list
    monitor_adj, _ = graph_builder.monitor_graph(pressure_sensor_ids)
    if flow_sensor_ids:
        flow_graph = graph_builder.cal_g(flow_sensor_ids, pressure_sensor_ids)
    else:
        flow_graph = np.zeros((max(1, len(pressure_sensor_ids)), len(pressure_sensor_ids)), dtype=np.float32)
    return {
        "graph_builder": graph_builder,
        "pressure_sensor_ids": pressure_sensor_ids,
        "flow_sensor_ids": flow_sensor_ids,
        "monitor_adj": torch.tensor(monitor_adj, dtype=torch.float32, device=device),
        "flow_graph": torch.tensor(flow_graph, dtype=torch.float32, device=device),
    }


def _build_base_model(model_name: str, task: str, graph_data: dict, device: torch.device):
    in_features = _resolve_pressure_feature_count() or len(graph_data["pressure_sensor_ids"])
    num_classes = _resolve_num_classes(task)
    model_name = _normalize_model_name(model_name)
    fusion_mode = dict(_get_value("FUSION_MODES", default={}) or {}).get(model_name, "cascade")

    if model_name == "GRU":
        return GRUClassifier(in_features, _get_value("GRU_HIDDEN", default=32), _get_value("LINEAR_HIDDEN", default=64), num_classes, num_layers=int(_get_value("GRU_NUM_LAYERS", default=3)), dropout=float(_get_value("GRU_DROPOUT", default=0.0))).to(device)
    if model_name == "CNN":
        return CNNClassifier(_get_value("IN_CHANNELS", default=1), _get_value("HIDDEN_CHANNEL1", default=8), _get_value("HIDDEN_CHANNEL2", default=16), _get_value("OUT_CHANNEL", default=32), _get_value("HIDDEN_FEATURES", default=64), num_classes).to(device)
    if model_name == "GCN":
        model = GCNClassifier(
            in_features=in_features,
            gcn_hidden=_get_value("GCN_HIDDEN", "GRU_HIDDEN", default=32),
            linear_hidden=_get_value("LINEAR_HIDDEN", default=64),
            out_features=num_classes,
            dropout=_get_value("DROPOUT_RATE", default=0.0),
            node_readout=_get_value("NODE_READOUT", default="mean"),
            time_readout=_get_value("TIME_READOUT", default="mean"),
            readout_order=_get_value("READOUT_ORDER", default="node_time"),
            classifier_mode=_get_value("CLASSIFIER_MODE", default="pooled"),
            feature_norm=_get_value("FEATURE_NORM", default="none"),
            use_residual=_get_value("USE_RESIDUAL", default=False),
            temporal_conv_kernel=_get_value("TEMPORAL_CONV_KERNEL", default=0),
            temporal_conv_layers=_get_value("TEMPORAL_CONV_LAYERS", default=0),
            temporal_hidden=_get_value("TEMPORAL_HIDDEN", default=0),
            temporal_layers=_get_value("TEMPORAL_LAYERS", default=1),
            temporal_dropout=_get_value("TEMPORAL_DROPOUT", default=0.0),
            temporal_readout=_get_value("TEMPORAL_READOUT", default="last"),
            temporal_input=_get_value("TEMPORAL_INPUT", default="flatten"),
            adj=graph_data["monitor_adj"],
        )
        return model.to(device)
    if model_name == "FA-DENSENET":
        return FADenseNet(in_features, num_classes, _get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60), block_dropout=float(_get_value("FA_BLOCK_DROPOUT", default=0.5))).to(device)

    if model_name == "FF-GRU":
        return FFGRUClassifier(
            graph_data["flow_graph"],
            graph_data["monitor_adj"],
            in_features,
            _get_value("GRU_HIDDEN", default=32),
            _get_value("LINEAR_HIDDEN", default=64),
            _get_value("LAQ_HIDDEN_DIM", default=max(1, in_features)),
            num_classes,
            sequence_length=_get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60),
            num_layers=int(_get_value("GRU_NUM_LAYERS", default=3)),
            dropout=float(_get_value("GRU_DROPOUT", default=0.0)),
            fusion_mode=fusion_mode,
        ).to(device)
    if model_name == "FF-CNN":
        return FFCNNClassifier(
            graph_data["flow_graph"],
            graph_data["monitor_adj"],
            len(graph_data["pressure_sensor_ids"]),
            _get_value("IN_CHANNELS", default=1),
            _get_value("HIDDEN_CHANNEL1", default=8),
            _get_value("HIDDEN_CHANNEL2", default=16),
            _get_value("OUT_CHANNEL", default=32),
            _get_value("HIDDEN_FEATURES", default=64),
            num_classes,
            sequence_length=_get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60),
            fusion_mode=fusion_mode,
        ).to(device)
    if model_name == "FF-GCN":
        backbone_adj = (torch.ones_like(graph_data["monitor_adj"])
                        if fusion_mode == "dense_topology" else graph_data["monitor_adj"])
        return FFGCNClassifier(
            graph_data["flow_graph"],
            graph_data["monitor_adj"],
            len(graph_data["pressure_sensor_ids"]),
            _get_value("GCN_HIDDEN", "GRU_HIDDEN", default=32),
            _get_value("LINEAR_HIDDEN", default=64),
            _get_value("LAQ_HIDDEN_DIM", default=max(1, in_features)),
            num_classes,
            backbone_adj,
            dropout=_get_value("DROPOUT_RATE", default=0.0),
            node_readout=_get_value("NODE_READOUT", default="mean"),
            time_readout=_get_value("TIME_READOUT", default="mean"),
            readout_order=_get_value("READOUT_ORDER", default="node_time"),
            classifier_mode=_get_value("CLASSIFIER_MODE", default="pooled"),
            feature_norm=_get_value("FEATURE_NORM", default="none"),
            use_residual=_get_value("USE_RESIDUAL", default=False),
            temporal_conv_kernel=_get_value("TEMPORAL_CONV_KERNEL", default=0),
            temporal_conv_layers=_get_value("TEMPORAL_CONV_LAYERS", default=0),
            temporal_hidden=_get_value("TEMPORAL_HIDDEN", default=0),
            temporal_layers=_get_value("TEMPORAL_LAYERS", default=1),
            temporal_dropout=_get_value("TEMPORAL_DROPOUT", default=0.0),
            temporal_readout=_get_value("TEMPORAL_READOUT", default="last"),
            temporal_input=_get_value("TEMPORAL_INPUT", default="flatten"),
            sequence_length=_get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60),
            fusion_mode=fusion_mode,
        ).to(device)
    if model_name == "FF-FA-DENSENET":
        return FFFADenseNetClassifier(
            graph_data["flow_graph"],
            graph_data["monitor_adj"],
            len(graph_data["pressure_sensor_ids"]),
            in_features,
            num_classes,
            _get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60),
            fusion_mode=fusion_mode,
            block_dropout=float(_get_value("FA_BLOCK_DROPOUT", default=0.5)),
        ).to(device)

    raise ValueError(f"Unsupported model name: {model_name}")


def _detection_dataset(split: str, fusion: bool):
    prefix = split.upper()
    length = _get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60)
    options = dict(need_time_len=length, left_offset=_get_value("LEFT_OFFSET", default=0),
                   right_offset=_get_value("RIGHT_OFFSET", default=0),
                   norm_id_file=_get_value(f"{prefix}_NORM_IDS"), burst_id_file=_get_value(f"{prefix}_BURST_IDS"))
    if fusion:
        return MyAlarmDataQ(
            _get_value(f"{prefix}_NORM_PRESSURE", f"{prefix}_NORM_FILE"),
            _get_value(f"{prefix}_NORM_FLOW"),
            _get_value(f"{prefix}_BURST_PRESSURE", f"{prefix}_BURST_FILE"),
            _get_value(f"{prefix}_BURST_FLOW"),
            flow_reduce=_get_value("FLOW_REDUCE", default="sum"),
            flow_interval_seconds=float(_get_value("FLOW_INTERVAL_SECONDS", default=60.0)), **options)
    return MyAlarmData(_get_value(f"{prefix}_NORM_FILE"), _get_value(f"{prefix}_BURST_FILE"), **options)


def _make_detection_loaders(fusion: bool):
    batch_size = int(_get_value("BATCH_SIZE", default=32))
    _require_prepartitioned_data()

    return (DataLoader(_detection_dataset("train", fusion), batch_size=batch_size, shuffle=True),
            [DataLoader(_detection_dataset("val", fusion), batch_size=batch_size)],
            [DataLoader(_detection_dataset("test", fusion), batch_size=batch_size)])


def _make_localization_dataset(fusion: bool):
    sequence_length = _get_value("SEQUENCE_LENGTH", "SERIES_LENGTH", "SERIOUS_LEN", default=60)
    if fusion:
        return MyLocationDataQ(
            _get_value("TRAIN_DATA_PATH", "TRAIN_PRESSURE_PATH"),
            _get_value("TRAIN_LABEL_PATH"),
            _get_value("TRAIN_FLOW_PATH"),
            need_time_len=sequence_length,
            left_offset=_get_value("LEFT_OFFSET", default=0),
            right_offset=_get_value("RIGHT_OFFSET", default=0),
            flow_reduce=_get_value("FLOW_REDUCE", default="sum"),
            flow_interval_seconds=float(_get_value("FLOW_INTERVAL_SECONDS", default=60.0)),
            sample_id_file=_get_value("TRAIN_BURST_IDS"),
        )
    return MyLocationData(
        _get_value("TRAIN_DATA_PATH", "TRAIN_PRESSURE_PATH"),
        _get_value("TRAIN_LABEL_PATH"),
        sample_id_file=_get_value("TRAIN_BURST_IDS"),
        need_time_len=sequence_length,
        left_offset=_get_value("LEFT_OFFSET", default=0),
        right_offset=_get_value("RIGHT_OFFSET", default=0),
    )


def _localization_loaders(split: str, fusion: bool, batch_size: int):
    prefix = split.upper()
    pressures = list(_get_value(f"{prefix}_PRESSURE_PATHS", default=[]))
    labels = list(_get_value(f"{prefix}_LABEL_PATHS", default=[]))
    flows = list(_get_value(f"{prefix}_FLOW_PATHS", default=[]))
    ids = list(_get_value(f"{prefix}_BURST_ID_PATHS", default=[]))
    if not pressures or len(pressures) != len(labels):
        raise ValueError(f"{prefix} requires matching nonempty PRESSURE_PATHS and LABEL_PATHS")
    if fusion and len(flows) != len(pressures):
        raise ValueError(f"{prefix}_FLOW_PATHS must match PRESSURE_PATHS")
    if ids and len(ids) != len(pressures):
        raise ValueError(f"{prefix}_BURST_ID_PATHS must match PRESSURE_PATHS")
    options = dict(need_time_len=_get_value("SEQUENCE_LENGTH", default=60),
                   left_offset=_get_value("LEFT_OFFSET", default=0),
                   right_offset=_get_value("RIGHT_OFFSET", default=0))
    loaders = []
    for index, pressure in enumerate(pressures):
        if fusion:
            dataset = MyLocationDataQ(pressure, labels[index], flows[index],
                         flow_reduce=_get_value("FLOW_REDUCE", default="sum"),
                         flow_interval_seconds=float(_get_value("FLOW_INTERVAL_SECONDS", default=60.0)),
                         sample_id_file=ids[index] if ids else None, **options)
        else:
            dataset = MyLocationData(pressure, labels[index],
                         sample_id_file=ids[index] if ids else None, **options)
        loaders.append(DataLoader(dataset, batch_size=batch_size))
    return loaders


def _make_localization_loaders(fusion: bool):
    batch_size = int(_get_value("BATCH_SIZE", default=32))
    _require_prepartitioned_data()
    dataset = _make_localization_dataset(fusion)
    return (DataLoader(dataset, batch_size=batch_size, shuffle=True),
            _localization_loaders("val", fusion, batch_size),
            _localization_loaders("test", fusion, batch_size))


def _move_inputs(inputs, device: torch.device):
    if isinstance(inputs, (tuple, list)):
        return tuple(t.to(device) for t in inputs)
    return inputs.to(device)


def _run_epoch(model, loader, loss_fn, device: torch.device, optimizer=None, collect_outputs=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_sample_ids = []
    dataset_indices = getattr(loader.dataset, "indices", None)
    base_dataset = getattr(loader.dataset, "dataset", loader.dataset)
    dataset_sample_ids = getattr(base_dataset, "sample_id", None)

    for inputs, targets in loader:
        inputs = _move_inputs(inputs, device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        if collect_outputs:
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
            if dataset_indices is not None:
                start = total_samples - batch_size
                selected = list(dataset_indices[start:total_samples])
                if dataset_sample_ids is not None:
                    all_sample_ids.extend([str(dataset_sample_ids[i]) for i in selected])
                else:
                    all_sample_ids.extend(selected)
            elif dataset_sample_ids is not None:
                start = total_samples - batch_size
                all_sample_ids.extend([str(dataset_sample_ids[i]) for i in range(start, total_samples)])
            else:
                all_sample_ids.extend(range(total_samples - batch_size, total_samples))

    result = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }
    if collect_outputs:
        result.update({"targets": np.asarray(all_targets), "predictions": np.asarray(all_predictions), "sample_id": np.asarray(all_sample_ids)})
    return result


def _save_prediction_rows(path: Path, metrics: dict) -> None:
    """Save local prediction details for later CAC/sample-level auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "y_true", "y_pred"])
        writer.writerows(zip(metrics["sample_id"], metrics["targets"], metrics["predictions"]))


def _build_loss(task: str, num_classes: int, device: torch.device):
    weights = _get_value("DETECTION_CLASS_WEIGHTS" if task == "detection" else "LOCALIZATION_CLASS_WEIGHTS",
                         "CLASS_WEIGHTS", default=[])
    if weights:
        weight = torch.tensor(weights, dtype=torch.float32, device=device)
        if weight.numel() != num_classes:
            raise ValueError(f"CLASS_WEIGHTS needs {num_classes} entries, got {weight.numel()}")
    else:
        weight = None
    return torch.nn.CrossEntropyLoss(weight=weight)


def _train(task: str, model_name: str):
    device = _device()
    _set_seed(int(_get_value("SEED", default=0)))
    epochs = int(_get_value("MAX_EPOCHS", default=0))
    if epochs < 1:
        raise ValueError("Set a positive MAX_EPOCHS in the local config.py before training")
    graph_data = _build_graphs(device)
    fusion = _normalize_model_name(model_name).startswith("FF-")
    if fusion and not graph_data["flow_sensor_ids"]:
        raise ValueError("Feature-fusion models require FLOW_SENSOR_INDEXES in config.py")
    model = _build_base_model(model_name, task, graph_data, device)
    loss_fn = _build_loss(task, _resolve_num_classes(task), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(_get_value("LEARNING_RATE", default=1e-3)))

    if task == "detection":
        train_loader, val_loaders, test_loaders = _make_detection_loaders(fusion)
    else:
        train_loader, val_loaders, test_loaders = _make_localization_loaders(fusion)

    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"]}
        validation = []
        for index, val_loader in enumerate(val_loaders, start=1):
            val_metrics = _run_epoch(model, val_loader, loss_fn, device, optimizer=None, collect_outputs=True)
            validation.append((len(val_loader.dataset), val_metrics["loss"]))
            prefix = f"val{index}_"
            row.update({prefix + "loss": val_metrics["loss"], prefix + "accuracy": val_metrics["accuracy"]})
            if task == "detection":
                row.update({prefix + key: value for key, value in detection_metrics(val_metrics["targets"], val_metrics["predictions"]).items()})
            else:
                row.update({prefix + key: value for key, value in localization_metrics(val_metrics["targets"], val_metrics["predictions"]).items()})
        mean_val_loss = sum(count * loss for count, loss in validation) / sum(count for count, _ in validation)
        row["val_loss"] = mean_val_loss
        if mean_val_loss < best_loss:
            best_loss, best_epoch = mean_val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(row)

    model.load_state_dict(best_state)
    test_report = {"best_epoch": best_epoch, "best_val_loss": best_loss}
    for index, test_loader in enumerate(test_loaders, start=1):
        metrics = _run_epoch(model, test_loader, loss_fn, device, collect_outputs=True)
        prefix = f"test{index}_"
        test_report.update({prefix + "loss": metrics["loss"], prefix + "accuracy": metrics["accuracy"]})
        task_metrics = (detection_metrics if task == "detection" else localization_metrics)(metrics["targets"], metrics["predictions"])
        test_report.update({prefix + key: value for key, value in task_metrics.items()})
        if bool(_get_value("SAVE_PREDICTIONS", default=False)):
            configured = str(_get_value("PREDICTIONS_PATH", default="outputs/{task}_{model}_test{index}_predictions.csv"))
            prediction_path = Path(configured.format(task=task, model=model_name.replace("-", "_"), index=index))
            _save_prediction_rows(prediction_path, metrics)

    if bool(_get_value("SAVE_MODEL", default=False)):
        checkpoint_path = Path(_get_value("MODEL_SAVE_PATH", default="outputs/model_state.pth"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    if bool(_get_value("SAVE_RESULTS", default=False)):
        results_path = Path(_get_value("RESULTS_PATH", "RESULT_FILE_PATH", default="outputs/results.csv"))
        results_path.parent.mkdir(parents=True, exist_ok=True)
        header = list(dict.fromkeys(key for row in history + [test_report] for key in row))
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(history + [test_report])

    return history


def main():
    task = str(_get_value("TASK", default="detection")).lower()
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported TASK: {task}. Supported values: {supported}")
    model_name = _normalize_model_name(str(_get_value("MODEL_NAME", "MODEL_TYPE", default="GRU")))
    _train(task, model_name)


if __name__ == "__main__":
    main()
