# coding=utf-8
"""Dataset and graph utilities."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: Path = BASE_DIR) -> str:
    """Resolve a local path against the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _sample_ids(pressure_file, count, base_dir, id_file=None, kind="burst"):
    """Use shared event IDs or source-specific IDs without inferring cross-file matches."""
    if id_file is not None:
        ids = pd.read_csv(resolve_path(id_file, base_dir), header=None, dtype=str).iloc[:, 0].to_numpy()
    else:
        source = resolve_path(pressure_file, base_dir)
        ids = np.asarray([f"{kind}:{source}:{i}" for i in range(count)])
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("sample IDs must be unique and match the number of windows")
    return ids


def _window_bounds(need_time_len: int, time_series_len: int, left_offset: int = 0, right_offset: int = 0):
    start = int((time_series_len - need_time_len) / 2)
    left = start + left_offset
    right = time_series_len - start + right_offset
    return left, right


def _reshape_sequences(data: np.ndarray, time_series_len: int) -> np.ndarray:
    if data.shape[0] % time_series_len != 0:
        raise ValueError(
            f"row count {data.shape[0]} is not divisible by time_series_len={time_series_len}"
        )
    return data.reshape(-1, time_series_len, data.shape[1])


def _windowed_series(
    data: np.ndarray,
    need_time_len: int,
    time_series_len: int,
    left_offset: int = 0,
    right_offset: int = 0,
) -> np.ndarray:
    left, right = _window_bounds(need_time_len, time_series_len, left_offset, right_offset)
    seq = _reshape_sequences(data, time_series_len)[:, left:right, :]
    return np.ascontiguousarray(seq.astype(np.float32, copy=False))


def _windowed_flow_sums(
    data: np.ndarray,
    need_time_len: int,
    time_series_len: int,
    left_offset: int = 0,
    right_offset: int = 0,
) -> np.ndarray:
    left, right = _window_bounds(need_time_len, time_series_len, left_offset, right_offset)
    seq = _reshape_sequences(data, time_series_len)[:, left:right, :]
    summed = seq.sum(axis=1)
    return np.ascontiguousarray(summed.astype(np.float32, copy=False))


def _windowed_flow(
    data: np.ndarray,
    need_time_len: int,
    time_series_len: int,
    left_offset: int = 0,
    right_offset: int = 0,
    flow_reduce: str = "sum",
    flow_interval_seconds: float = 60.0,
) -> np.ndarray:
    left, right = _window_bounds(need_time_len, time_series_len, left_offset, right_offset)
    seq = _reshape_sequences(data, time_series_len)[:, left:right, :]
    mode = str(flow_reduce).lower()
    if mode == "sum":
        # Flow files contain interval readings; sum them within each event window.
        seq = seq.sum(axis=1)
    elif mode == "volume":
        if flow_interval_seconds <= 0:
            raise ValueError("flow_interval_seconds must be positive")
        # Input rates are in volume/second; the window represents total volume.
        seq = seq.sum(axis=1) * flow_interval_seconds
    elif mode != "sequence":
        raise ValueError(f"unsupported flow_reduce: {flow_reduce}")
    return np.ascontiguousarray(seq.astype(np.float32, copy=False))


class CreateGraph:
    """Construct topological graph representations from hydraulic network models."""

    def __init__(self, wn):
        self.wn = wn

    def reverse_pipes(self):
        """Identify pipes whose dominant simulated flow reverses their stored direction."""

        sim = wntr.sim.EpanetSimulator(self.wn)
        result = sim.run_sim()
        self.wn.reset_initial_values()
        flow = result.link["flowrate"].loc[:, self.wn.pipe_name_list]
        return flow.columns[(flow < 0).sum(axis=0) > (flow > 0).sum(axis=0)].tolist()

    def wsn_graph(self, remove_reservoirs: bool = True):
        """Construct a directed graph and its adjacency matrix."""

        pipes = set(self.reverse_pipes())
        g = self.wn.get_graph()
        for source, target, key in list(g.edges(keys=True)):
            if key in pipes:
                attributes = dict(g.get_edge_data(source, target, key))
                g.remove_edge(source, target, key)
                g.add_edge(target, source, key=key, **attributes)

        if remove_reservoirs:
            for i in range(self.wn.num_nodes - self.wn.num_reservoirs, self.wn.num_nodes):
                g.remove_node(self.wn.node_name_list[i])

        return g, np.asarray(nx.adjacency_matrix(g).todense()).astype(np.float32)

    def knn_graph(self):
        """Construct a 2-hop directed connectivity graph."""

        g, adj = self.wsn_graph()
        nodes = list(g.nodes)

        for n, node in enumerate(nodes):
            for v in g.successors(node):
                adj[n, nodes.index(v)] = 1
                for sv in g.successors(v):
                    adj[n, nodes.index(sv)] = 1
            for u in g.predecessors(node):
                adj[n, nodes.index(u)] = 1
                for su in g.predecessors(u):
                    adj[n, nodes.index(su)] = 1

        return g, adj.astype(np.float32)

    def monitor_graph(self, monitors):
        """Construct adjacency matrix for monitor nodes and return their positions."""

        g, _ = self.wsn_graph()
        pos = nx.get_node_attributes(g, "pos")
        sensor_positions = [pos[node] for node in monitors]

        graph = np.zeros((len(monitors), len(monitors)))
        for i, src in enumerate(monitors):
            for j, target in enumerate(monitors):
                if i != j and nx.has_path(g, src, target):
                    path = nx.dijkstra_path(g, src, target)
                    if len(set(monitors) & set(path)) <= 2:
                        graph[i, j] = 1
        return graph.astype(np.float32), sensor_positions

    def cal_g(self, flow_sensors, pressure_sensors):
        """Compute connectivity matrix between flow and pressure sensors."""

        graph, _ = self.wsn_graph(remove_reservoirs=False)
        connectivity = np.zeros((len(flow_sensors), len(pressure_sensors)), dtype=np.float32)

        for i, flow_node in enumerate(flow_sensors):
            for j, pressure_node in enumerate(pressure_sensors):
                if nx.has_path(graph, flow_node, pressure_node):
                    connectivity[i, j] = 1
        return connectivity

    def index2id(self, indexs):
        """Convert numerical indices to node IDs."""

        return [self.wn.node_name_list[i] for i in indexs]


class MyLocationData(Dataset):
    """Dataset for pipe burst localization."""

    def __init__(
        self,
        data_file,
        label_file,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        is_norm: bool = False,
        base_dir: Path = BASE_DIR,
        sample_id_file=None,
    ):
        self.data = pd.read_csv(resolve_path(data_file, base_dir), header=None).values
        if is_norm:
            self.data = StandardScaler().fit_transform(self.data)
        self.data = _windowed_series(
            self.data,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        self.labels = pd.read_csv(resolve_path(label_file, base_dir), header=None, index_col=0).values
        self.labels = self.labels.reshape(-1).astype(np.int64) - 1
        self.sample_id = _sample_ids(data_file, self.labels.shape[0], base_dir, sample_id_file)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


class MyLocationBigData(Dataset):
    """Streaming variant of the localization dataset."""

    def __init__(
        self,
        data_file,
        label_file,
        samples,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        base_dir: Path = BASE_DIR,
    ):
        self.data = pd.read_csv(resolve_path(data_file, base_dir), header=None, iterator=True)
        self.label = pd.read_csv(resolve_path(label_file, base_dir), header=None, index_col=0, iterator=True)
        self.samples = samples
        self.need_time_len = need_time_len
        self.time_series_len = time_series_len
        self.left_offset = left_offset
        self.right_offset = right_offset

    def __getitem__(self, item):
        x = self.data.get_chunk(self.time_series_len).values
        y = self.label.get_chunk(1).values[0] - 1
        start = int((self.time_series_len - self.need_time_len) / 2)
        x = x[start + self.left_offset : self.time_series_len - start + self.right_offset, :]
        return x.astype(np.float32), y.astype(np.int64)

    def __len__(self):
        return self.samples


class MyLocationDataQ(Dataset):
    """Localization dataset with flow measurements."""

    def __init__(
        self,
        data_file,
        label_file,
        flow_file,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        is_norm: bool = False,
        flow_reduce: str = "sum",
        base_dir: Path = BASE_DIR,
        sample_id_file=None,
        flow_interval_seconds: float = 60.0,
    ):
        self.data = pd.read_csv(resolve_path(data_file, base_dir), header=None).values
        if is_norm:
            self.data = StandardScaler().fit_transform(self.data)
        self.qdata = pd.read_csv(resolve_path(flow_file, base_dir), header=None).values
        self.data = _windowed_series(
            self.data,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        self.qdata = _windowed_flow(
            self.qdata,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
            flow_reduce=flow_reduce,
            flow_interval_seconds=flow_interval_seconds,
        )
        self.label = pd.read_csv(resolve_path(label_file, base_dir), header=None, index_col=0).values
        self.label = self.label.reshape(-1).astype(np.int64) - 1
        self.sample_id = _sample_ids(data_file, self.label.shape[0], base_dir, sample_id_file)

    def __getitem__(self, item):
        return (self.data[item], self.qdata[item]), self.label[item]

    def __len__(self):
        return self.label.shape[0]


class MyLocationBigDataQ(Dataset):
    """Streaming localization dataset with flow measurements."""

    def __init__(
        self,
        data_file,
        label_file,
        flow_file,
        samples,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        flow_reduce: str = "sum",
        base_dir: Path = BASE_DIR,
    ):
        self.data = pd.read_csv(resolve_path(data_file, base_dir), header=None, iterator=True)
        self.label = pd.read_csv(resolve_path(label_file, base_dir), header=None, index_col=0, iterator=True)
        self.flow = pd.read_csv(resolve_path(flow_file, base_dir), header=None, iterator=True)
        self.samples = samples
        self.need_time_len = need_time_len
        self.time_series_len = time_series_len
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.flow_reduce = str(flow_reduce).lower()

    def __getitem__(self, item):
        x = self.data.get_chunk(self.time_series_len).values
        q = self.flow.get_chunk(self.time_series_len).values
        y = self.label.get_chunk(1).values[0] - 1
        start = int((self.time_series_len - self.need_time_len) / 2)
        x = x[start + self.left_offset : self.time_series_len - start + self.right_offset, :]
        q = q[start + self.left_offset : self.time_series_len - start + self.right_offset, :]
        if self.flow_reduce == "sum":
            q = q.sum(0)
        elif self.flow_reduce != "sequence":
            raise ValueError(f"unsupported flow_reduce: {self.flow_reduce}")
        return (x.astype(np.float32), q.astype(np.float32)), y.astype(np.int64)

    def __len__(self):
        return self.samples


class MyAlarmData(Dataset):
    """Binary alarm detection dataset."""

    def __init__(
        self,
        norm_data_file,
        burst_data_file,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        is_norm: bool = False,
        is_balance: bool = True,
        base_dir: Path = BASE_DIR,
        norm_id_file=None,
        burst_id_file=None,
    ):
        norm = pd.read_csv(resolve_path(norm_data_file, base_dir), header=None).values
        burst = pd.read_csv(resolve_path(burst_data_file, base_dir), header=None).values
        samples_norm = int(norm.shape[0] / time_series_len)
        samples_burst = int(burst.shape[0] / time_series_len)
        self.samples = np.array([0] * samples_norm + [1] * samples_burst).astype(np.int64)
        self.sample_id = np.concatenate((
            _sample_ids(norm_data_file, samples_norm, base_dir, norm_id_file, kind="normal"),
            _sample_ids(burst_data_file, samples_burst, base_dir, burst_id_file),
        ))
        self.data = np.concatenate((norm, burst), axis=0).astype(np.float32)
        if is_norm:
            self.data = StandardScaler().fit_transform(self.data)
        self.data = _windowed_series(
            self.data,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
        )

    def __getitem__(self, item):
        return self.data[item], self.samples[item]

    def __len__(self):
        return len(self.samples)


class MyAlarmDataQ(Dataset):
    """Binary alarm detection dataset with flow measurements."""

    def __init__(
        self,
        norm_pdata_file,
        norm_qdata_file,
        burst_pdata_file,
        burst_qdata_file,
        need_time_len: int = 60,
        time_series_len: int = 60,
        left_offset: int = 0,
        right_offset: int = 0,
        is_norm: bool = False,
        flow_reduce: str = "sum",
        base_dir: Path = BASE_DIR,
        norm_id_file=None,
        burst_id_file=None,
        flow_interval_seconds: float = 60.0,
    ):
        norm_pdata = pd.read_csv(resolve_path(norm_pdata_file, base_dir), header=None).values
        norm_qdata = pd.read_csv(resolve_path(norm_qdata_file, base_dir), header=None).values
        burst_pdata = pd.read_csv(resolve_path(burst_pdata_file, base_dir), header=None).values
        burst_qdata = pd.read_csv(resolve_path(burst_qdata_file, base_dir), header=None).values
        samples_norm = int(norm_pdata.shape[0] / time_series_len)
        samples_burst = int(burst_pdata.shape[0] / time_series_len)
        self.samples = np.array([0] * samples_norm + [1] * samples_burst).astype(np.int64)
        self.sample_id = np.concatenate((
            _sample_ids(norm_pdata_file, samples_norm, base_dir, norm_id_file, kind="normal"),
            _sample_ids(burst_pdata_file, samples_burst, base_dir, burst_id_file),
        ))
        self.data = np.concatenate((norm_pdata, burst_pdata), axis=0).astype(np.float32)
        if is_norm:
            self.data = StandardScaler().fit_transform(self.data)
        self.qdata = np.concatenate((norm_qdata, burst_qdata), axis=0).astype(np.float32)
        self.data = _windowed_series(
            self.data,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        self.qdata = _windowed_flow(
            self.qdata,
            need_time_len,
            time_series_len,
            left_offset=left_offset,
            right_offset=right_offset,
            flow_reduce=flow_reduce,
            flow_interval_seconds=flow_interval_seconds,
        )

    def __getitem__(self, item):
        return (self.data[item], self.qdata[item]), self.samples[item]

    def __len__(self):
        return len(self.samples)
