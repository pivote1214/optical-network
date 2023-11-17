from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


def gen_one_demand_size(
    generation_principle: str, 
    min_size: int, 
    max_size: int
    ) -> int:
    """
    minsizeからmax_sizeまでのサイズで，1つの通信要求のサイズをランダムに生成する
    """
    if generation_principle == 'Random':
        return int(np.random.randint(min_size, max_size))
    if generation_principle == 'Gaussian':
        return int(np.random.normal(min_size, max_size))


def gen_all_demands_offline(
    generation_principle: str, 
    G: nx.Graph, 
    N_DEMANDS: int, 
    min_size: int, 
    max_size: int, 
    seed: int = None
    ) -> dict:
    """N_DEMANDSの数だけ通信要求を生成する"""
    nodes = list(G.nodes)
    demands = {}

    if seed is not None:
        np.random.seed(seed)
    
    for i in range(1, N_DEMANDS+1):
        source, destination = np.random.choice(nodes, 2, replace=False)
        size = gen_one_demand_size(generation_principle, min_size, max_size)
        demands[i] = (source, destination, size)

    return demands
