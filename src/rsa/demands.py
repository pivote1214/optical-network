import random
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Demand:
    id: int
    source: Any
    target: Any
    traffic_vol: float
    arrival_time: float
    holding_time: float


def gen_dynamic_demands(
    graph: nx.Graph, 
    n_demands: int, 
    traffic_vol_population: list[float], 
    holding_time_ave: int, 
    erlang: int, 
    seed: Optional[int] = None
    ) -> list[Demand]:
    if seed is not None:
        random.seed(seed)
    
    nodes = list(graph.nodes)
    arrival_rate = erlang / holding_time_ave
    demands = []
    arrival_time = 0
    for id in range(1, n_demands+1):
        source, target = random.sample(nodes, 2)
        traffic_volume = random.choice(traffic_vol_population)
        holding_time = random.expovariate(1 / holding_time_ave)
        arrival_time += random.expovariate(arrival_rate)
        demands.append(Demand(id, source, target, traffic_volume, arrival_time, holding_time))

    return demands
    

def _gen_demand_size(
    demands_population: list[int], 
    ) -> int:
    """
    generate one demand size from demands_population
    """
    return np.random.choice(demands_population)


def gen_all_demands_offline(
    graph: nx.Graph, 
    n_demands: int, 
    demands_population: list[int], 
    seed: int = None
    ) -> dict:
    """gemerate all demands offline"""
    nodes = list(graph.nodes)
    demands = {}

    if seed is not None:
        np.random.seed(seed)
    
    for i in range(1, n_demands+1):
        source, destination = np.random.choice(nodes, 2, replace=False)
        size = _gen_demand_size(demands_population)
        demands[i] = (source, destination, size)

    return demands
