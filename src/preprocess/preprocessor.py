from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from src.utils.graph import calc_path_length

class DataPreprocessor:
    def __init__(
        self, 
        graph: nx.Graph, 
        all_paths: Dict[Tuple[int, int], List[int]],
        demands: Dict[int, Tuple[int, int, int]], 
        slices: List[int], 
        width: Dict[str, int] = {"OC": 37.5, "GB": 6.25, "FS": 12.5}, 
        traffic_bpsk: float = 50
        ):
        self.graph = graph
        self.all_paths = all_paths
        self.demands = demands
        self.slices = slices
        self.width = width
        self.traffic_bpsk = traffic_bpsk


    def preprocess(self) -> None:
        """
        前処理を行うメソッド
        """
        path_set = self.make_paths()
        num_slots, channels = self.make_channels(path_set)

        return path_set, num_slots, channels

        
    def make_paths(self) -> Dict[int, List[List[int]]]:
        """
        要求に対するパスの集合を作成する関数
        """
        path_set = {}
        for d_ind in self.demands.keys():
            source, destination, _ = self.demands[d_ind]
            path_set[d_ind] = self.all_paths[source, destination]

        return path_set


    def make_channels(
        self, 
        path_set: Dict[int, List[List[int]]]
        ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[List[int]]] ]:
        """
        通信要求に対するチャネルの集合を作成する関数
        """
        max_slot = len(self.slices)
        num_slots = {}
        channels = {}
        for d_ind, demand in self.demands.items():
            for p_ind, path in enumerate(path_set[d_ind]):
                # 変調方式の選択
                path_length = calc_path_length(self.graph, path)
                modulation_format = self._select_modulation_format(path_length)
                required_slots = self._calc_required_slots(demand[2], modulation_format)
                
                num_slots[d_ind, p_ind] = required_slots
                channels[d_ind, p_ind] = self._calc_candidate_channel(required_slots, max_slot)

        return num_slots, channels

    
    def _calc_candidate_channel(self, slot_num: int, max_slot: int) -> List[List[int]]:
        """
        使用するスロット数と最大スロット数が与えられたときに、候補チャネルを列挙する関数
        """
        channels = []
        # 候補チャネルの列挙
        for i in range(max_slot - slot_num + 1):
            channel = []
            for j in range(slot_num):
                channel.append(i + j)
            channels.append(channel)

        return channels


    # 候補パスが与えられたときに変調方式を決定し、スペクトル効率を計算する関数
    def _select_modulation_format(self, path_length: int) -> int:
        """
        変調方式を決定し、スペクトル効率を計算する関数
        """
        if path_length <= 600:
            modulation_format = 4
        elif path_length <= 1200:
            modulation_format = 3
        elif path_length <= 3500:
            modulation_format = 2
        elif path_length <= 6300:
            modulation_format = 1
        else:
            raise ValueError("Path length is too long.")

        return modulation_format


    def _calc_required_slots(
        self, 
        demand_size: float, 
        modulation_format: int
        ) -> int:
        """
        必要なスロット数を計算する関数
        """
        required_slots = np.ceil(
            np.ceil(demand_size / modulation_format * self.traffic_bpsk) * \
                self.width['OC'] + 2 * self.width['GB'] 
                / self.width['FS']
                )

        return required_slots
