import pickle

import numpy as np

from src.graph import calc_path_weight, is_edge_in_path, select_modulation_format
from src.rsa.demands import gen_all_demands_offline
from src.rsa.path_channel.lower_bound import PathLowerBoundInput
from src.rsa.path_channel.params import Parameter, Width


def make_input_lower(params: Parameter) -> PathLowerBoundInput:
    """Generate input for lower bound model"""
    # make edge, slot and demand
    E = {e_ind: edge for e_ind, edge in enumerate(params.graph.edges)}
    S = list(range(params.num_slots))
    D = gen_all_demands_offline(
        params.graph,
        params.num_demands,
        demands_population=params.demands_population,
        seed=params.demands_seed,
    )
    # load all paths
    all_paths = params.all_paths
    # make path, num_slots, channel and delta
    P = _make_path(D, all_paths)
    num_slots = _make_num_slots(params, D, P)
    delta = _calculate_delta(E, D, P)
    input = PathLowerBoundInput(
        E=E,
        S=S,
        D=D,
        P=P,
        num_slots=num_slots,
        delta=delta,
        result_dir=params.result_dir,
        demand_seed=params.demands_seed,
        timelimit=params.timelimit.lower,
    )

    return input


def make_channels(
    S: list[int], num_slots: dict[tuple[int, int], int]
) -> dict[tuple[int, int], list[list[int]]]:
    """Generate channel set"""
    max_slot = len(S)
    channels = {}
    for d_ind, p_ind in num_slots.keys():
        channels[d_ind, p_ind] = _calc_candidate_channel(
            num_slots[d_ind, p_ind], max_slot
        )

    return channels


def calculate_gamma(
    S: list[int],
    D: dict[int, tuple[int, int, int]],
    P: dict[int, list[list[int]]],
    C: dict[tuple[int, int], list[list[int]]],
) -> dict[tuple[int, int, int, int], int]:
    """calculate gamma (channel contains slice or not)"""
    gamma = {}
    for d_ind, _ in D.items():
        for p_ind, _ in enumerate(P[d_ind]):
            for c_ind, channel in enumerate(C[d_ind, p_ind]):
                for s_ind, slice in enumerate(S):
                    gamma[d_ind, p_ind, c_ind, s_ind] = 1 if slice in channel else 0

    return gamma


def _make_path(
    D: dict[int, tuple[int, int, int]],
    all_paths: dict[tuple[int, int], list[list[int]]],
) -> dict[int, list[list[int]]]:
    """Generate path set"""
    path_set = {}
    for d_ind in D.keys():
        source, destination, _ = D[d_ind]
        path_set[d_ind] = all_paths[source, destination]

    return path_set


def _make_num_slots(
    params: Parameter, D: dict[int, tuple[int, int, int]], P: dict[int, list[list[int]]]
) -> tuple[dict[tuple[int, int], int]]:
    """Generate channel set"""
    num_slots = {}
    for d_ind, demand in D.items():
        for p_ind, path in enumerate(P[d_ind]):
            # select modulation format
            path_length = calc_path_weight(params.graph, path)
            modulation_format = select_modulation_format(path_length)
            # calculate required slots
            required_slots = calc_required_slots(
                demand[2], modulation_format, params.width, params.TRAFFIC_BPSK
            )
            num_slots[d_ind, p_ind] = required_slots

    return num_slots


def _calc_candidate_channel(slot_num: int, max_slot: int) -> list[list[int]]:
    """Calculate candidate channels"""
    channels = []
    for i in range(max_slot - slot_num + 1):
        channel = []
        for j in range(slot_num):
            channel.append(i + j)
        channels.append(channel)

    return channels


def calc_required_slots(
    demand_size: float, modulation_format: int, width: Width, TRAFFIC_BPSK: float
) -> int:
    """Calculate required slots"""
    required_slots = np.ceil(
        (
            np.ceil((demand_size / (modulation_format * TRAFFIC_BPSK))) * width.optical_carrier
            + 2 * width.guard_band
        )
        / width.frequency_slot
    )
    required_slots = int(required_slots)

    return required_slots


def _calculate_delta(
    E: dict[int, tuple[int, int]],
    D: dict[int, tuple[int, int, int]],
    P: dict[int, list[list[int]]],
) -> dict[tuple[int, int, int], int]:
    """calculate delta (path contains edge or not)"""
    delta = {}
    for e_ind, edge in E.items():
        for d_ind, _ in D.items():
            for p_ind, path in enumerate(P[d_ind]):
                delta[e_ind, d_ind, p_ind] = is_edge_in_path(path, edge)

    return delta
