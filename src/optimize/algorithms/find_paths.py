import pickle

from src.utils.paths import PATHS_DIR
from src.optimize.algorithms.optical_network import OpticalNetwork


def k_shortest_paths(
    optical_network: OpticalNetwork, 
    k: int, 
    source: int, 
    target: int
    ) -> list[list[int]]:
    """Find k shortest paths between source and target"""
    with open(PATHS_DIR / optical_network.graph_name / f"kSP_k={k}_alpha={None}.pickle", "rb") as f:
        paths_info = pickle.load(f)
        paths = paths_info["paths"]

    return paths[source, target]
