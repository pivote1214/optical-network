import networkx as nx
from copy import deepcopy

from src.optimize.algorithms.optical_network import OpticalNetwork
from src.utils.graph import calc_path_length

def repeat_dijkstra(
    network: nx.DiGraph, 
    k: int, 
    beta: int = 10
    ) -> dict[tuple[int, int], list[list[int]]]:
    """Find k paths between all pairs of nodes by repeating dijkstra"""
    G = deepcopy(network)
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    all_paths = {(u, v): [] for u in nodes for v in nodes if u != v}
    edge_usage = {(u, v): 0 for u, v in edges}
    weight_range = max([G[u][v]["weight"] for u, v in edges]) - min([G[u][v]["weight"] for u, v in edges])

    # repeat dijkstra
    times = 0
    while times < k:
        times += 1
        for v in nodes:
            from_v = nx.single_source_dijkstra_path(G, v)
            for u in nodes:
                if v == u:
                    continue
                all_paths[v, u].append(from_v[u])
                for s, t in zip(from_v[u], from_v[u][1:]):
                    edge_usage[s, t] += 1
        # operate weight
        max_usage = max(edge_usage.values())
        min_usage = min(edge_usage.values())
        for u, v in edges:
            G[u][v]["weight"] += \
                (edge_usage[u, v] - min_usage) / (max_usage - min_usage) * weight_range * beta

    for key, paths in all_paths.items():
        all_paths[key] = list(set(tuple(path) for path in paths))
        print(all_paths[key])
        all_paths[key] = [path for path in all_paths[key] if calc_path_length(network, path) <= 6300]

    return all_paths
