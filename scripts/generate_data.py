"""Hydraulic data-generation entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import wntr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.create_data import CreateData
from data.operate_inp import OperateModel


def _network_file() -> str:
    network_file = getattr(config, "NETWORK_FILE", None) or getattr(config, "MODEL_FILE", None) or getattr(config, "INP_FILE", None)
    if not network_file:
        raise ValueError("NETWORK_FILE, MODEL_FILE, or INP_FILE must be set in config.py")
    return str(network_file)


def prepare_hydraulic_model() -> Path:
    wn = wntr.network.WaterNetworkModel(_network_file())
    operator = OperateModel(
        wn,
        old_step=getattr(config, "ORIG_PATTERN_STEP", 3600),
        new_step=getattr(config, "NEW_PATTERN_STEP", 60),
    )
    operator.configure_and_save()
    return Path(getattr(config, "MODIFIED_MODEL_PATH", "outputs/modified_model.inp"))


def generate_normal_data(network_file=None):
    wn = wntr.network.WaterNetworkModel(str(network_file or _network_file()))
    creator = CreateData(
        wn,
        burst_level=getattr(config, "BURST_LEVEL", [0.01, 0.02]),
        burst_interval=getattr(config, "BURST_INTERVAL", 3600),
        discharge_coeff=getattr(config, "DISCHARGE_COEFF", 0.75),
    )
    creator.normal_data()


def generate_burst_data(network_file=None):
    wn = wntr.network.WaterNetworkModel(str(network_file or _network_file()))
    creator = CreateData(
        wn,
        burst_level=getattr(config, "BURST_LEVEL", [0.01, 0.02]),
        burst_interval=getattr(config, "BURST_INTERVAL", 3600),
        discharge_coeff=getattr(config, "DISCHARGE_COEFF", 0.75),
    )
    creator.pipe_burst(start_time=getattr(config, "DEFAULT_BURST_START", 0))


def generate_sensitivity_data(network_file=None):
    wn = wntr.network.WaterNetworkModel(str(network_file or _network_file()))
    creator = CreateData(
        wn,
        burst_level=getattr(config, "BURST_LEVEL", [0.01, 0.02]),
        burst_interval=getattr(config, "BURST_INTERVAL", 3600),
        discharge_coeff=getattr(config, "DISCHARGE_COEFF", 0.75),
    )
    creator.sensitive(
        bursts=getattr(config, "SENSITIVITY_BURST_LEVEL", 0.2),
        start_time=getattr(config, "DEFAULT_BURST_START", 0),
    )


def main():
    actions_enabled = [
        getattr(config, "BUILD_MODIFIED_MODEL", False),
        getattr(config, "GENERATE_NORMAL_DATA", False),
        getattr(config, "GENERATE_BURST_DATA", False),
        getattr(config, "GENERATE_SENSITIVITY_DATA", False),
    ]
    if not any(actions_enabled):
        raise ValueError(
            "No data-generation step is enabled. Set one or more of "
            "BUILD_MODIFIED_MODEL, GENERATE_NORMAL_DATA, GENERATE_BURST_DATA, "
            "or GENERATE_SENSITIVITY_DATA in config.py."
        )
    network_file = prepare_hydraulic_model() if getattr(config, "BUILD_MODIFIED_MODEL", False) else _network_file()
    if getattr(config, "GENERATE_NORMAL_DATA", False):
        generate_normal_data(network_file)
    if getattr(config, "GENERATE_BURST_DATA", False):
        generate_burst_data(network_file)
    if getattr(config, "GENERATE_SENSITIVITY_DATA", False):
        generate_sensitivity_data(network_file)


if __name__ == "__main__":
    main()
