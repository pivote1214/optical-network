import copy
import heapq
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.rsa.demands import Demand
from src.rsa.greedy.spectrum_allocation import first_fit
from src.rsa.optical_network import OpticalNetwork, Width

__all__ = ["simulate_online_rsa"]


def simulate_online_rsa(
    graph: nx.DiGraph,
    num_slots: int,
    demands: list[Demand],
    all_paths: dict[tuple[Any, Any], list[list[Any]]],
    modulation_formats: list[tuple[int, int]],
    width: Width,
    traffic_vol_bpsk: int,
    ) -> tuple[dict[Demand, tuple[list[Any], int, int]], list[Demand], list[dict[tuple[Any, Any], NDArray[np.bool_]]]]:
    optical_network = OpticalNetwork(
        graph, 
        num_slots, 
        modulation_formats, 
        width, 
        traffic_vol_bpsk
        )

    # イベントの管理
    events: list[tuple[float, str, Demand]] = []
    for demand in demands:
        heapq.heappush(events, (demand.arrival_time, "2-arrival", demand))
    allocated_demands: dict[Demand, tuple[list[Any], int, int]] = {}
    blocked_demands: list[Demand] = []

    optical_net_snap = list()
    section_time = 1
    while events:
        event_time, event_type, demand = heapq.heappop(events)
        # 記録
        if event_time >= section_time:
            optical_net_snap.append(copy.deepcopy(optical_network.available))
            section_time += 1
        # 到着イベント
        if event_type == "2-arrival":
            paths = all_paths[demand.source, demand.target]
            path_assigned, starting_slot, ending_slot = first_fit(
                optical_network, 
                paths, 
                demand
                )

            if path_assigned is None:
                blocked_demands.append(demand)
            else:
                # 割り当てられた場合は解放イベントをスケジュール
                release_time = event_time + demand.holding_time
                heapq.heappush(events, (release_time, "1-release", demand))
                allocated_demands[demand] = (path_assigned, starting_slot, ending_slot)
                optical_network.assign_slots(path_assigned, starting_slot, ending_slot)
        # 解放イベント
        elif event_type == "1-release":
            path_released, starting_slot, ending_slot = allocated_demands[demand]
            optical_network.release_slots(path_released, starting_slot, ending_slot)

    return allocated_demands, blocked_demands, optical_net_snap
