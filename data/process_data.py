"""Data processing and augmentation utilities for hydraulic experiments."""

import re
import os
import wntr
import numpy as np
import pandas as pd
from copy import deepcopy
from threading import Thread
from multiprocessing import Pool
import config  # Import configuration settings


def nodes_by_diameter(wn, diameter=400, operator='=='):
    """
    Filters nodes based on connecting pipe diameters.

    Args:
        wn: wntr.network.WaterNetworkModel
        diameter: Pipe diameter threshold (mm)
        operator: Filter condition ('==', '<=', '>=')

    Returns:
        list: Filtered node IDs
    """
    nodes = []
    pipe_names = set(wn.pipe_name_list)

    for pipe_name in pipe_names:
        pipe = wn.get_link(pipe_name)
        dia_m = diameter / 1000  # Convert mm to meters

        if operator == '==' and pipe.diameter == dia_m:
            nodes.extend([pipe.start_node_name, pipe.end_node_name])
        elif operator == '<=' and pipe.diameter <= dia_m:
            nodes.extend([pipe.start_node_name, pipe.end_node_name])
        elif operator == '>=' and pipe.diameter >= dia_m:
            nodes.extend([pipe.start_node_name, pipe.end_node_name])

    return list(set(nodes))  # Remove duplicates


def partition():
    """
    Creates node-to-partition mapping from monitoring scheme.

    Returns:
        dict: {node_id: partition_id}
    """
    # Load water network model
    wn = wntr.network.WaterNetworkModel(config.MODEL_FILE)

    # Load partitioning scheme
    partitions = pd.read_excel(
        config.PARTITION_SCHEME_FILE,
        sheet_name=config.SHEET_NAME
    )

    # Create node ID to partition ID mapping
    node_partition_map = {}
    for _, row in partitions.iterrows():
        node_id = wn.node_name_list[row.iloc[0]]
        partition_id = row.iloc[1]
        node_partition_map[node_id] = partition_id

    return node_partition_map


def process_norm_data(sensors, pfname, pname, qfname, qname, is_saveq=True):
    """
    Processes normal operation data with optional flow data.

    Args:
        sensors: List of sensor indices
        pfname: Input pressure data file
        pname: Output pressure data file
        qfname: Input flow data file
        qname: Output flow data file
        is_saveq: Whether to process flow data
    """
    # Read pressure data
    pdf = pd.read_csv(pfname, header=0, index_col=0)

    # Read flow data if enabled
    if is_saveq:
        qdf = pd.read_csv(qfname, header=0, index_col=0)

    # Process in sliding windows
    for k in range(config.NORMAL_SAMPLES):
        # Extract pressure segment
        pressure_segment = pdf.iloc[k:k + config.WINDOW_SIZE, sensors]
        pressure_segment.to_csv(pname, header=False, index=False, mode='a')

        # Extract flow segment
        if is_saveq:
            flow_segment = qdf.iloc[k:k + config.WINDOW_SIZE, :]
            flow_segment.to_csv(qname, header=False, index=False, mode='a')


def process_augmented_normal(sensors, normal_data_file, output_file):
    """
    Augments normal data with Gaussian noise.

    Args:
        sensors: List of sensor indices
        normal_data_file: Path to normal data file
        output_file: Output file path
    """
    # Read original data
    pdf = pd.read_csv(normal_data_file, header=0, index_col=0)

    # Noise-based augmentation
    for i in range(config.AUGMENTATION_REPEATS):
        for j in range(config.SUB_SAMPLES):
            # Generate noise
            noise = np.random.normal(0, config.NOISE_STD,
                                     size=(config.WINDOW_SIZE, len(sensors)))

            # Apply noise and save
            noisy_data = pdf.iloc[j:j + config.WINDOW_SIZE, sensors].values + noise
            pd.DataFrame(noisy_data).to_csv(
                output_file,
                header=False,
                index=False,
                mode='a'
            )


class ProcessData:
    """Organizes burst data for machine learning pipelines."""

    def __init__(self, sensors, partitions, time_series_len=60):
        """
        Args:
            sensors: List of sensor indices
            partitions: Node-to-partition mapping {node: partition_id}
            time_series_len: Time window length (default=60)
        """
        self.sensors = sensors
        self.partitions = partitions
        self.data_folder = config.BURST_DATA_FOLDER
        self.time_series_len = time_series_len

    def _filter_files(self, pattern, burst_levels, diameter_nodes=None):
        """
        Filters burst data files by criteria.

        Returns:
            tuple: (pressure_files, flow_files, leak_files, node_ids)
        """
        all_files = os.listdir(self.data_folder)
        pressure_files = []
        flow_files = []
        leak_files = []
        node_ids = []

        for filename in all_files:
            match = re.match(pattern, filename)
            if not match:
                continue

            burst_level = float(match.group(2))
            node_id = match.group(1)

            # Apply filters
            if burst_level not in burst_levels:
                continue
            if diameter_nodes and node_id not in diameter_nodes:
                continue

            # Validate files
            pressure_files.append(filename)
            flow_files.append(filename.replace('P', 'Q_all'))
            leak_file = filename.replace('P', 'Q')
            if os.path.exists(os.path.join(self.data_folder, leak_file)):
                leak_files.append(leak_file)
            node_ids.append(node_id)

        return pressure_files, flow_files, leak_files, node_ids

    def process_pressure(self, input_files, node_list, output_data, output_labels):
        """
        Processes pressure data files.

        Args:
            input_files: List of input file names
            node_list: Node IDs corresponding to input_files
            output_data: Output CSV file for data
            output_labels: Output CSV file for labels
        """
        for filename, node_id in zip(input_files, node_list):
            print(f'Processing: {filename}')
            file_path = os.path.join(self.data_folder, filename)

            # Read and process data
            df = pd.read_csv(file_path, header=0, index_col=0)
            df_processed = df.iloc[:, self.sensors]

            # Save to dataset
            df_processed.to_csv(
                output_data,
                header=False,
                index=False,
                mode='a'
            )

            # Generate labels
            partition_id = self.partitions.get(node_id, -1)
            samples = len(df) // self.time_series_len
            labels = [[node_id, partition_id]] * samples

            # Save labels
            pd.DataFrame(labels).to_csv(
                output_labels,
                header=False,
                index=False,
                mode='a'
            )

    def process_flow(self, flow_files, output_file, leak_files=None, leak_output=None):
        """
        Processes flow data files.

        Args:
            flow_files: List of flow data files
            output_file: Output file path
            leak_files: List of leak flow files (optional)
            leak_output: Leak output file path (optional)
        """
        for filename in flow_files:
            print(f'Processing flow: {filename}')
            file_path = os.path.join(self.data_folder, filename)
            df = pd.read_csv(file_path, header=0, index_col=0)
            df.to_csv(output_file, header=False, index=False, mode='a')

        # Process leak flow if requested
        if leak_files and leak_output:
            for filename in leak_files:
                print(f'Processing leak: {filename}')
                file_path = os.path.join(self.data_folder, filename)
                df = pd.read_csv(file_path, header=0, index_col=0)
                df.to_csv(leak_output, header=False, index=False, mode='a')

    def parallel_process(self, pressure_files, node_ids, flow_files,
                         output_pressure, output_label, output_flow,
                         leak_files=None, output_leak=None):
        """
        Parallel processing of pressure and flow data.
        """
        # Pressure processing thread
        pressure_thread = Thread(
            target=self.process_pressure,
            args=(pressure_files, node_ids, output_pressure, output_label)
        )

        # Flow processing thread
        flow_thread = Thread(
            target=self.process_flow,
            args=(flow_files, output_flow, leak_files, output_leak)
        )

        # Start and manage threads
        pressure_thread.daemon = True
        flow_thread.daemon = True
        pressure_thread.start()
        flow_thread.start()
        pressure_thread.join()
        flow_thread.join()
        print(f'Data processed: {output_pressure} | {output_flow}')


def multiprocessing_pipeline(parameters):
    """
    Multiprocessing wrapper for data processing.

    Args:
        parameters: Processing configuration dictionary.
    """
    # Initialize data processor
    processor = ProcessData(
        sensors=parameters['sensors'],
        partitions=parameters['partitions'],
        time_series_len=parameters.get('time_series_len', 60)
    )

    # Filter files
    pfiles, qfiles, lqfiles, node_ids = processor._filter_files(
        pattern=parameters['pattern'],
        burst_levels=parameters['burst_levels'],
        diameter_nodes=parameters.get('nodes_by_diameter')
    )

    # Process data
    processor.parallel_process(
        pressure_files=pfiles,
        node_ids=node_ids,
        flow_files=qfiles,
        output_pressure=parameters['pressure_output'],
        output_label=parameters['label_output'],
        output_flow=parameters['flow_output'],
        leak_files=lqfiles,
        output_leak=parameters.get('leak_output')
    )


if __name__ == '__main__':
    print("Import ProcessData and call process_data with local settings.")
