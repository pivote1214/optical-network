import os
import sys
sys.path.append(os.path.abspath('../../'))

import networkx as nx
import numpy as np


def _gen_one_demand_size(
    demands_population: list[int], 
    ) -> int:
    """
    generate one demand size from demands_population
    """
    return np.random.choice(demands_population)


def gen_all_demands_offline(
    graph: nx.Graph, 
    N_DEMANDS: int, 
    demands_population: list[int], 
    seed: int = None
    ) -> dict:
    """gemerate all demands offline"""
    nodes = list(graph.nodes)
    demands = {}

    if seed is not None:
        np.random.seed(seed)
    
    for i in range(1, N_DEMANDS+1):
        source, destination = np.random.choice(nodes, 2, replace=False)
        size = _gen_one_demand_size(demands_population)
        demands[i] = (source, destination, size)

    return demands

if __name__ == "__main__":
    import pickle

    from utils.namespaces import TEST_DIR
    from utils.network import load_network

    network_names = ['JPN12', 'NSF', 'EURO16', 'GRID2x2', 'GRID2x3', 'GRID3x3', 'GRID3x4', 'GRID4x4']
    for network_name in network_names:
        graph = load_network(network_name)
        n_demands = 20
        populations = [50, 100, 150, 200]
        for seed in range(2, 21, 2):
            demands = gen_all_demands_offline(graph, n_demands, populations, seed)
            if not os.path.exists(os.path.join(TEST_DIR, "demands", network_name)):
                os.makedirs(os.path.join(TEST_DIR, "demands", network_name))
            with open(os.path.join(TEST_DIR, "demands", network_name, f"seed={seed}.pkl"), "wb") as f:
                pickle.dump(demands, f)
