"""Hydraulic network preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wntr
from scipy.interpolate import interp1d

import config


class OperateModel:
    """Configure hydraulic parameters and resample demand patterns."""

    def __init__(self, wn, old_step: int = 3600, new_step: int = 60):
        self.wn = wn
        self.old_step = old_step
        self.new_step = new_step

    def _config_hydraulic_params(self):
        hyd_options = self.wn.options.hydraulic
        hyd_options.demand_model = "PDD"
        hyd_options.required_pressure = config.REQ_PRESSURE
        hyd_options.minimum_pressure = config.MIN_PRESSURE
        hyd_options.pressure_exponent = config.PRESSURE_EXPONENT

    def _config_time_params(self):
        time_options = self.wn.options.time
        time_options.duration = config.SIM_DURATION
        time_options.hydraulic_timestep = config.HYD_TIMESTEP
        time_options.pattern_timestep = config.PATTERN_TIMESTEP
        time_options.report_timestep = config.REPORT_TIMESTEP

    def _interpolate_patterns(self):
        for pattern_name in self.wn.pattern_name_list:
            pattern = self.wn.get_pattern(pattern_name)
            x = np.arange(0, (len(pattern) + 1) * self.old_step, self.old_step)
            y = list(pattern.multipliers)
            y.append(pattern.multipliers[0])
            interpolator = interp1d(x, y, kind="quadratic")

            nx = np.arange(0, len(pattern) * self.old_step, self.new_step)
            ny = interpolator(nx)

            if config.ADD_PATTERN_NOISE:
                noise = 1.0 - np.random.normal(loc=0, scale=config.NOISE_STD, size=ny.shape)
                ny *= noise

            pattern.multipliers = ny.tolist()

    def configure_and_save(self):
        self._config_hydraulic_params()
        self._config_time_params()
        self._interpolate_patterns()

        output_path = Path(config.MODIFIED_MODEL_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.wn.write_inpfile(filename=str(output_path), version=2.2)
        print(f"Modified model saved to: {output_path}")


if __name__ == "__main__":
    wn = wntr.network.WaterNetworkModel(config.MODEL_FILE)
    OperateModel(
        wn,
        old_step=config.ORIG_PATTERN_STEP,
        new_step=config.NEW_PATTERN_STEP,
    ).configure_and_save()
