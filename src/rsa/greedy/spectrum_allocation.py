from typing import Any, Optional

import numpy as np

from src._utils import calc_required_slots
from src.rsa.demands import Demand
from src.rsa.optical_network import OpticalNetwork


def first_fit(
    optical_network: OpticalNetwork, paths: list[list[Any]], demand: Demand
) -> tuple[Optional[list[Any]], Optional[int], Optional[int]]:
    """First fitアルゴリズムの実装

    Parameters
    ----------
    optical_network: OpticalNetwork
        Optical network
    paths: list[list[Any]]
        Candidate paths
    demand: Demand
        Demand

    Returns
    -------
    tuple[Optional[list[Any]], Optional[int], Optional[int]]
        Path assigned, starting slot, ending slot
    """
    best_path, best_starting_slot, best_ending_slot = None, np.inf, None
    for path in paths:
        required_slots = calc_required_slots(
            demand,
            path,
            optical_network.graph,
            optical_network.modulation_formats,
            optical_network.width,
            optical_network.traffic_vol_bpsk,
        )
        path_availability: np.ndarray = optical_network.get_path_availability(path)
        starting_slot = None
        for idx in range(len(path_availability) - required_slots + 1):
            if np.all(path_availability[idx : idx + required_slots]):
                starting_slot = idx
                break
        if starting_slot is None:
            continue
        if starting_slot < best_starting_slot:
            best_path = path
            best_starting_slot = starting_slot
            best_ending_slot = starting_slot + required_slots - 1

    if best_path is None:
        return None, None, None
    else:
        return best_path, best_starting_slot, best_ending_slot
