import networkx as nx
from dataclasses import dataclass, field


@dataclass(frozen=False)
class Parameter:
    network_name:       str
    graph:              nx.Graph
    num_slots:          int
    num_demands:        int
    demands_population: int
    demands_seed:       int
    k:                  int
    path_algo_name:     str
    alpha:              float
    bound_algo:         str
    TIMELIMIT:          int = 3600
    W:                  dict[str, float] = field(default_factory=lambda: {"OC": 37.5, "GB": 6.25, "FS": 12.5})
    TRAFFIC_BPSK:       float = 50
