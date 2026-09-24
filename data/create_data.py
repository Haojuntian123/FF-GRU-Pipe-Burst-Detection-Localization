"""Hydraulic data-generation utilities."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wntr

import config


class CreateData:
    """Generate hydraulic simulation datasets for local experiments."""

    def __init__(
        self,
        wn,
        burst_level=None,
        duration=None,
        burst_interval=None,
        discharge_coeff=None,
    ):
        self.wn = wn
        self.duration = duration if duration is not None else self.wn.options.time.duration
        self.burst_level = np.array(burst_level if burst_level is not None else config.BURST_LEVEL)
        self.burst_interval = burst_interval if burst_interval is not None else config.BURST_INTERVAL
        self.discharge_coeff = discharge_coeff if discharge_coeff is not None else config.DISCHARGE_COEFF

    @staticmethod
    def _pressure_noise(frame):
        """Apply the configured measurement noise to simulated pressures."""
        std = float(getattr(config, "PRESSURE_NOISE_STD", 0.0))
        if std <= 0:
            return frame
        noisy = frame.copy()
        noisy.iloc[:, :] += np.random.normal(0.0, std, size=noisy.shape)
        return noisy

    def normal_data(self, is_save: bool = True, pressure_file=None, flow_file=None):
        """Simulate the normal operating condition and optionally save it to disk."""

        pressure_file = pressure_file or config.NORMAL_PRESSURE_FILE
        flow_file = flow_file or config.NORMAL_FLOW_FILE

        sim = wntr.sim.WNTRSimulator(self.wn)
        result = sim.run_sim()
        self.wn.reset_initial_values()
        pressure = self._pressure_noise(result.node["pressure"].iloc[:-1, :-3])
        flow = -result.node["demand"].iloc[:-1, -3:]
        if is_save:
            pressure_path = Path(pressure_file)
            flow_path = Path(flow_file)
            pressure_path.parent.mkdir(parents=True, exist_ok=True)
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            pressure.to_csv(pressure_path)
            flow.to_csv(flow_path)
        return pressure, flow

    def _get_diameter(self, node):
        """Return the maximum pipe diameter connected to a junction node."""

        diameters = [self.wn.get_link(link).diameter for link in self.wn.get_links_for_node(node)]
        return max(diameters) if diameters else 0.0

    def _burst(self, node, area, start_time):
        """Simulate a single burst event for one node and one start time."""

        self.wn.options.time.duration = start_time + self.burst_interval
        junction = self.wn.get_node(node)
        junction.remove_leak(self.wn)
        junction.add_leak(
            self.wn,
            area=area,
            discharge_coeff=self.discharge_coeff,
            start_time=start_time,
            end_time=start_time + self.burst_interval,
        )
        sim = wntr.sim.WNTRSimulator(self.wn)
        result = sim.run_sim()
        self.wn.reset_initial_values()
        pressure_data = self._pressure_noise(result.node["pressure"].iloc[-61:-1, :-3])
        leak_flow = result.node["leak_demand"][node].iloc[-61:-1]
        reservoir_flow = -result.node["demand"].iloc[-61:-1, -3:]
        return pressure_data, leak_flow, reservoir_flow

    def pipe_burst(self, start_time=None, nodes=None):
        """Generate burst simulation datasets for the selected nodes."""

        start_time = config.DEFAULT_BURST_START if start_time is None else start_time
        node_list = nodes or self.wn.junction_name_list
        output_dir = Path(config.BURST_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        for node in node_list:
            diameter = self._get_diameter(node)
            areas = np.pi * (diameter**2) / 4 * self.burst_level
            for index, area in enumerate(areas):
                current_start = start_time
                pressure_frames = pd.DataFrame()
                leak_frames = pd.DataFrame()
                flow_frames = pd.DataFrame()

                while current_start <= (self.duration - self.burst_interval):
                    pressure_data, leak_data, flow_data = self._burst(node, area, current_start)
                    pressure_frames = pd.concat([pressure_frames, pressure_data])
                    leak_frames = pd.concat([leak_frames, leak_data])
                    flow_frames = pd.concat([flow_frames, flow_data])
                    current_start += self.burst_interval

                severity = f"{self.burst_level[index]:.2f}"
                pressure_frames.to_csv(output_dir / f"P_{node}_{severity}.csv")
                leak_frames.to_csv(output_dir / f"Q_{node}_{severity}.csv")
                flow_frames.to_csv(output_dir / f"Q_all_{node}_{severity}.csv")
                print(f"{node}_{severity} completed")

            np.savetxt(output_dir / f"{node}_burst_area.csv", areas, fmt="%.6f", delimiter=",")

        self.wn.reset_initial_values()

    def sensitive(self, bursts, start_time=None):
        """Compute a sensitivity matrix for the specified burst severity."""

        start_time = config.DEFAULT_BURST_START if start_time is None else start_time
        try:
            normal_pressure = pd.read_csv(config.NORMAL_PRESSURE_FILE, header=0, index_col=0)
        except FileNotFoundError:
            normal_pressure, _ = self.normal_data(is_save=True)

        reference_pressure = normal_pressure.loc[start_time].values
        node_list = self.wn.junction_name_list
        sensitivity_matrix = np.zeros((0, len(node_list)))

        for node in node_list:
            diameter = self._get_diameter(node)
            area = np.pi * (diameter**2) / 4 * bursts
            burst_pressure, _, _ = self._burst(node, area, start_time)
            mid_idx = int(burst_pressure.shape[0] / 2)
            delta_p = reference_pressure - burst_pressure.values[mid_idx, :]
            base_drop = reference_pressure[node_list.index(node)] - burst_pressure.values[mid_idx, node_list.index(node)]
            sensitivity = delta_p / base_drop
            sensitivity_matrix = np.vstack([sensitivity_matrix, sensitivity])

        output_dir = Path(config.SENSITIVITY_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sensitive_{bursts}_{start_time // 3600}.csv"
        np.savetxt(output_path, sensitivity_matrix, fmt="%.4f", delimiter=",")
        self.wn.reset_initial_values()


def mulcreatedata(wn, burst_level, nodes):
    """Multiprocessing helper for burst dataset generation."""

    model = CreateData(wn, burst_level=burst_level)
    model.pipe_burst(nodes=nodes)


if __name__ == "__main__":
    wn = wntr.network.WaterNetworkModel(config.INP_FILE)
    datamodel = CreateData(wn, burst_level=config.BURST_LEVEL)
    datamodel.normal_data()

    # Uncomment below to run full burst simulations.
    # datamodel.pipe_burst()

    # Uncomment below to run sensitivity analysis.
    # datamodel.sensitive(bursts=1.0, start_time=12 * 3600)

    # Uncomment below for multiprocessing burst generation.
    """
    node_list = wn.junction_name_list
    chunk_size = len(node_list) // 10 + 1
    pool = Pool(10)

    for i in range(10):
        node_chunk = node_list[i * chunk_size : (i + 1) * chunk_size]
        pool.apply_async(mulcreatedata, args=(wn, config.BURST_LEVEL, node_chunk))
        print(f"Process {i} started: {len(node_chunk)} nodes")

    pool.close()
    pool.join()
    print("All burst datasets generated")
    """
