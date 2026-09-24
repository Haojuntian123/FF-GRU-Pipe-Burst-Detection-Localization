"""Burst-event analysis utilities for hydraulic simulation outputs."""

import numpy as np
import pandas as pd
import wntr
import re
import os
from config import INP_FILE, DATA_DIR, OUTPUT_FILE, MONITORS, BURST_LEVEL_LEN  # Import configuration settings

class AnalyzeData:
    """
    Analyzes burst pipe data:
    -- Variables: a, burst level (severity)
    -- For each junction node:
          1. Pressure drop (min, median, mean, max) at monitoring points caused by burst.
          2. Burst flow (min, median, mean, max) at junction node.
          3. Pipe diameter of the junction node.
    Return:
        DataFrame columns: junction | diameter | burst_q | delta_pressure_min | delta_pressure_media | delta_pressure_mean | delta_pressure_max
    """
    def __init__(self, root=DATA_DIR, save_file=OUTPUT_FILE):
        """
        Initializes WaterNetworkModel, sets file paths and junction list.
        """
        self.wn = wntr.network.WaterNetworkModel(INP_FILE)
        self.junctions = self.wn.junction_name_list
        self.root = root
        self.save_file = save_file

    def index2id(self, indexs):
        """Converts integer indices to junction IDs."""
        ids = [self.junctions[i] for i in indexs]
        return ids

    def _get_datas(self, junction):
        """Gets data files for a junction, matches filenames by pattern P_junction_value.csv."""
        files = os.listdir(self.root)
        data = []
        burst_level = []
        for f in files:
            match = re.match(r'P_'+junction+r'_'+r'([0-9.]*)\.csv', f)
            if match:
                delta_p, index = self._analyze_pressure(self.root+'/'+match.group(0), junction)
                delta_q = self._analyze_flow(self.root+'/'+match.group(0).replace('P', 'Q'), index)
                pq = np.zeros((9,))
                pq[1::2] = delta_q
                pq[2::2] = delta_p
                pq[0] = max([self.wn.get_link(i).diameter for i in self.wn.get_links_for_node(junction)]) * 1000
                pq = list(pq)
                pq.insert(0, junction)
                data.append(pq)
                burst_level.append(float(match.group(1)))
        return data, burst_level

    def _get_monitors_datas(self, junction, monitors):
        """Gets data for specific monitors, similar to _get_datas but targeted to monitor points."""
        files = os.listdir(self.root)
        data = []
        burst_level = []
        for f in files:
            match = re.match(r'P_'+junction+r'_'+r'([0-9.]*)\.csv', f)
            if match:
                pq = np.zeros((len(monitors)+2,))
                delta_p, index = self._analyze_pressure(self.root+'/'+match.group(0), junction)
                delta_q = self._analyze_flow(self.root+'/'+match.group(0).replace('P', 'Q'), index)
                pq[0] = delta_q[-1]
                pq[1] = delta_p[-1]
                for i, n in enumerate(monitors,start=2):
                    delta_p, index = self._analyze_pressure(self.root + '/' + match.group(0), junction)
                    pq[i] = delta_p[-1]
                pq = list(pq)
                pq.insert(0, junction)
                data.append(pq)
                burst_level.append(float(match.group(1)))
        return data, burst_level

    def _analyze_pressure(self, file, junction):
        """Computes pressure drop from CSV file data."""
        df = pd.read_csv(file, header=0, index_col=0)
        data = df[junction].values
        delta = data[60::60] - data[30:-30:60]
        index = delta.argsort()[[0, int(len(delta)/2), -1]]
        return list(delta[index])+[delta.mean()], index

    def _analyze_flow(self, file, index):
        """Computes flow changes from CSV file data."""
        df = pd.read_csv(file, header=0, index_col=0)
        data = df.values
        delta = data[30:-30:60]
        return list(delta[index])+[delta.mean()]

    def analyze(self, burst_level_len=BURST_LEVEL_LEN):
        """Main method to analyze all junctions for general data. Outputs Excel with multiple sheets by burst level."""
        writer = pd.ExcelWriter(self.save_file)
        df = []
        sheet_name = []
        for i in range(burst_level_len):
            df.append([])
        for junction in self.junctions:
            data, burst_level = self._get_datas(junction)
            # self._analyze_junction(data)
            if data:
                if not sheet_name:
                    sheet_name = burst_level
                for m, n in enumerate(burst_level):
                    df[m].append(data[m])
        for i in range(burst_level_len):
            dataframe = pd.DataFrame(df[i], columns=[
                'junction', 'diameter',
                'burst_flow_min(m3/s)', 'burst_pressure_drop_min(m)',
                'burst_flow_median(m3/s)', 'burst_pressure_drop_median(m)',
                'burst_flow_max(m3/s)', 'burst_pressure_drop_max(m)',
                'burst_flow_mean(m3/s)', 'burst_pressure_drop_mean(m)'
            ])
            dataframe.to_excel(writer, sheet_name=str(sheet_name[i]))
        writer.close()

    def analyze_monitor(self, burst_level_len=BURST_LEVEL_LEN, monitors=[]):
        """Analyzes data for specific monitor nodes. Outputs Excel with multi-column data per burst level."""
        writer = pd.ExcelWriter(self.save_file)
        df = []
        sheet_name = []
        for i in range(burst_level_len):
            df.append([])
        for junction in self.junctions:
            data, burst_level = self._get_monitors_datas(junction, monitors)
            # self._analyze_junction(data)
            if data:
                if not sheet_name:
                    sheet_name = burst_level
                for m, n in enumerate(burst_level):
                    df[m].append(data[m])
        for i in range(burst_level_len):
            dataframe = pd.DataFrame(df[i], columns=[
                'junction', 'burst_flow_mean(m3/s)', 'burst_pressure_drop_mean(m)'] + [f'{i}_pressure_drop_mean(m)' for i in monitors]
                )
            dataframe.to_excel(writer, sheet_name=str(sheet_name[i]))
        writer.close()

if __name__ == '__main__':
    ids = AnalyzeData().index2id(MONITORS)  # Convert monitor indices to IDs using config
    model = AnalyzeData()  # Load defaults from config
    model.analyze_monitor(monitors=ids)  # Analyze with monitor-specific method
