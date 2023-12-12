from __future__ import annotations

from typing import Dict, List

import pickle
import numpy as np

from src.utils.paths import PATHS_DIR
from src.utils.graph import calc_path_length, is_edge_in_path
from src.demands.demands import gen_all_demands_offline
from src.optimize.models.RSA_PATH_LIMIT.model import Constant, IndexSet
from src.optimize.params import Parameter


def make_input(params: Parameter) -> tuple[IndexSet, Constant]:
    """Generate index set and constant"""
    # make index set
    E = {e_ind: edge for e_ind, edge in enumerate(params.graph.edges)}
    S = [s_ind for s_ind in range(params.num_slots)]
    D = gen_all_demands_offline(params.graph, params.num_demands, 
                                seed=params.demands_seed)
    all_paths_dir = \
        PATHS_DIR / params.network_name / f"{params.path_algo_name}_k={params.k}_alpha={params.alpha}.pickle"
    with open(all_paths_dir, mode="rb") as f:
        paths_info = pickle.load(f)
        all_paths = paths_info["paths"]
    P = _make_path(D, all_paths)
    n = _make_num_slots(params, S, D, P)
    index_set = IndexSet(E=E, S=S, D=D, P=P)

    # make constant
    delta = _calculate_delta(E, D, P)
    constant = Constant(num_slots=n, delta=delta, M=len(S))

    return index_set, constant


def _make_path(
    D: Dict[int, tuple[int, int, int]], 
    all_paths: Dict[tuple(int, int), List[List[int]]]
    ) -> Dict[int, List[List[int]]]:
    """Generate path set"""
    path_set = {}
    for d_ind in D.keys():
        source, destination, _ = D[d_ind]
        path_set[d_ind] = all_paths[source, destination]

    return path_set


def _make_num_slots(
    params: Parameter, 
    S: List[int], 
    D: Dict[int, tuple[int, int, int]], 
    P: Dict[int, List[List[int]]]
    ) -> tuple[Dict[tuple[int, int], int], Dict[tuple[int, int], List[List[int]]] ]:
    """Generate channel set"""
    max_slot = len(S)
    num_slots = {}
    for d_ind, demand in D.items():
        for p_ind, path in enumerate(P[d_ind]):
            # select modulation format
            path_length = calc_path_length(params.graph, path)
            modulation_format = _select_modulation_format(path_length)
            # calculate required slots
            required_slots = _calc_required_slots(demand[2], modulation_format, 
                                                  params.W, params.traffic_bpsk)
            num_slots[d_ind, p_ind] = required_slots

    return num_slots


def _select_modulation_format(path_length: int) -> int:
    """Select modulation format"""
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
    demand_size: float, 
    modulation_format: int, 
    W: dict[str, float], 
    traffic_bpsk: float
    ) -> int:
    """Calculate required slots"""
    # required_slots = np.ceil(
    #     np.ceil(
    #         (demand_size / (modulation_format * traffic_bpsk)) * \
    #         W['OC'] + 2 * W['GB']
    #         ) / W['FS']
    #         )
    required_slots = np.ceil(demand_size / (modulation_format * traffic_bpsk)) * 3 + 1
    required_slots = int(required_slots)

    return required_slots


def _calculate_delta(
    E: dict[int, tuple[int, int]], 
    D: dict[int, tuple[int, int, int]], 
    P: dict[int, List[List[int]]]
    ) -> Dict[tuple[int, int, int], int]:
    """calculate delta (path contains edge or not)"""
    delta = {}
    for e_ind, edge in E.items():
        for d_ind, _ in D.items():
            for p_ind, path in enumerate(P[d_ind]):
                delta[e_ind, d_ind, p_ind] = is_edge_in_path(path, edge)

    return delta
