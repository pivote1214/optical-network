import networkx as nx
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeLimit:
    lower: float
    upper: float
    main: float


@dataclass(frozen=True)
class Width:
    OC: float = 37.5
    GB: float = 6.25
    FS: float = 12.5


@dataclass(frozen=False)
class Parameter:
    network_name:       str
    graph:              nx.Graph
    num_slots:          int
    num_demands:        int
    demands_population: int
    demands_seed:       int
    paths_dir:          str
    result_dir:         str
    bound_algo:         str
    timelimit:          TimeLimit
    width:              Width
    TRAFFIC_BPSK:       float = 50
