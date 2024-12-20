from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.rsa.optical_network import Width

__all__ = ["TimeLimit", "Parameter"]


@dataclass(frozen=True)
class TimeLimit:
    lower: float
    upper: float
    main: float


@dataclass(frozen=False)
class Parameter:
    network_name:       str
    graph:              nx.Graph
    num_slots:          int
    num_demands:        int
    demands_population: int
    demands_seed:       int
    all_paths:          dict[tuple[Any, Any], list[list[Any]]]
    result_dir:         str
    bound_algo:         str
    timelimit:          TimeLimit
    width:              Width
    TRAFFIC_BPSK:       float = 50
