import networkx as nx
from dataclasses import dataclass, field


@dataclass(frozen=False)
class Parameter:
    network_name: str
    graph: nx.Graph
    num_slots: int
    num_demands: int
    demands_seed: int
    k: int
    path_algo_name: str
    alpha: float
    TimeLimit: int = 3600
    W: dict[str, float] = field(default_factory=lambda: {"OC": 37.5, "GB": 6.25, "FS": 12.5})
    traffic_bpsk: float = 50
