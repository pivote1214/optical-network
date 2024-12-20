import pickle

from src.namespaces import PATHS_DIR
from src.rsa.optical_network import OpticalNetwork


def k_shortest_paths(
    optical_network: OpticalNetwork, k: int, source: int, target: int
) -> list[list[int]]:
    """Find k shortest paths between source and target"""
    with open(
        PATHS_DIR / optical_network.graph_name / "kSP" / f"k={k}.pickle", "rb"
    ) as f:
        all_paths = pickle.load(f)

    return all_paths[source, target]


def k_shortest_paths_hop(
    optical_network: OpticalNetwork, k: int, source: int, target: int
) -> list[list[int]]:
    """Find k shortest paths (hop) between source and target"""
    with open(
        PATHS_DIR / optical_network.graph_name / "kSP-hop" / f"k={k}.pickle", "rb"
    ) as f:
        all_paths = pickle.load(f)

    return all_paths[source, target]


def kSPwLO(
    optical_network: OpticalNetwork, k: int, alpha: float, source: int, target: int
) -> list[list[int]]:
    """Find k shortest paths with link occupancy between source and target"""
    with open(
        PATHS_DIR
        / optical_network.graph_name
        / "kSPwLO"
        / f"k={k}_alpha={alpha}.pickle",
        "rb",
    ) as f:
        all_paths = pickle.load(f)

    return all_paths[source, target]


def repeat_dijkstra(
    optical_network: OpticalNetwork,
    k: int,
    beta: int,
    source: int,
    target: int,
) -> list[list[int]]:
    """Find all paths between source and target"""
    with open(
        PATHS_DIR
        / optical_network.graph_name
        / "Repeat Dijkstra"
        / f"k={k}_beta={beta}.pickle",
        "rb",
    ) as f:
        all_paths = pickle.load(f)

    return all_paths[source, target]
