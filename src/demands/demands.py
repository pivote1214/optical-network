from __future__ import annotations

from typing import List

import numpy as np
import networkx as nx


def _gen_one_demand_size(
    demands_population: List[int], 
    ) -> int:
    """
    generate one demand size from demands_population
    """
    return np.random.choice(demands_population)


def gen_all_demands_offline(
    graph: nx.Graph, 
    N_DEMANDS: int, 
    demands_population: List[int], 
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
