from dataclasses import dataclass
from typing import Any

import numpy as np
import networkx as nx

__all__ = ["OpticalNetwork", "Width"]


@dataclass(frozen=True)
class Width:
    optical_carrier: float
    guard_band: float
    frequency_slot: float


class OpticalNetwork:
    def __init__(
        self, 
        graph: nx.DiGraph, 
        num_slots: int, 
        modulation_formats: list[tuple[int, int]], 
        width: Width, 
        traffic_vol_bpsk: int
        ) -> None:
        self.graph = graph
        self.network_name = graph.name
        self.num_slots = num_slots
        self.modulation_formats = modulation_formats
        self.width = width
        self.traffic_vol_bpsk = traffic_vol_bpsk
        self.available = {(u, v): np.ones(num_slots, dtype=bool) for u, v in self.graph.edges}

    def assign_slots(
        self, 
        assigned_path: list[Any], 
        starting_slot: int, 
        ending_slot: int
        ) -> None:
        for u, v in zip(assigned_path[:-1], assigned_path[1:]):
            for s_ind in range(starting_slot, ending_slot + 1):
                self.available[u, v][s_ind] = False

    def release_slots(
        self, 
        released_path: list[Any], 
        starting_slot: int, 
        ending_slot: int
        ) -> None:
        for u, v in zip(released_path[:-1], released_path[1:]):
            for s_ind in range(starting_slot, ending_slot + 1):
                self.available[u, v][s_ind] = True

    def get_path_availability(
        self, 
        path: list[Any]
        ) -> np.ndarray:
        availability = np.ones(self.num_slots, dtype=bool)
        for u, v in zip(path[:-1], path[1:]):
            availability &= self.available[u, v]
        return availability

    def calc_used_rate(self) -> float:
        unused_slots = np.sum(np.array(list(self.available.values())))
        all_slots = self.num_slots * self.graph.number_of_edges()
        return (all_slots - unused_slots) / all_slots

    def calc_fragmentation_ratio(self) -> float:
        frag_ratio = 0
        for u, v in self.graph.edges:
            # Trueが連続する最大の長さを求める
            max_length = 0
            current_length = 0
            for slot in self.available[u, v]:
                if slot:
                    current_length += 1
                else:
                    max_length = max(max_length, current_length)
                    current_length = 0
            max_length = max(max_length, current_length)
            # フラグメンテーション比率を計算
            if sum(~self.available[u, v]) == self.num_slots:
                frag_ratio += 0
            else:
                frag_ratio += 1 - (max_length / (self.num_slots - sum(~self.available[u, v])))

        return frag_ratio / self.graph.number_of_edges()
