import pickle
import networkx as nx

from src.utils.paths import GRAPH_DIR
from src.optimize.algorithms.channel_assign import first_fit
from src.optimize.algorithms.find_paths import k_shortest_paths
from src.optimize.algorithms.optical_network import OpticalNetwork


def greedy_RMLSA_offline(
    graph_name: str, 
    num_slices: int, 
    k: int,
    demands: dict[int, tuple[int, int, int]]
    ) -> None: # TODO: return result dataclass
    """Greedy RMLSA algorithm for static traffic"""
    with open(GRAPH_DIR / f"{graph_name}.pickle", "rb") as f:
        graph: nx.DiGraph = pickle.load(f)
    
    optical_network = OpticalNetwork(graph, num_slices)
    for d_ind in demands:
        source, target, demand_size = demands[d_ind]
        paths_d = k_shortest_paths(optical_network, k, source, target)
        assined_slots = first_fit(optical_network, paths_d, demand_size)
        optical_network.renew(assined_slots)
