import numpy as np
import networkx as nx


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
    from src.utils.graph import load_network
    from src.utils.paths import DATA_DIR

    graph = load_network("NSF")
    n_demands = 5
    populations = [50, 100, 150, 200]
    for seed in range(1, 11):
        demands = gen_all_demands_offline(graph, n_demands, populations, seed*12)
        with open(f"{DATA_DIR}/input/demands/NSFNET/demands_{seed}_num={n_demands}_p={populations}.pickle", "wb") as f:
            pickle.dump(demands, f)
