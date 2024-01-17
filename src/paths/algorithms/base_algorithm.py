import time
import networkx as nx
from itertools import combinations


class BasePathAlgorithm:
    def __init__(
        self, 
        graph: nx.Graph, 
        path_nums: int, 
        length_limit: int = 6300, 
        ):
        self.graph = graph
        self.path_nums = path_nums
        self.length_limit = length_limit

    def find_path_pair(
        self, 
        source: int, 
        target: int
        ) -> tuple[list[tuple[int]], float]:
        """method to find k paths between source and target"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def find_all_paths(self) -> tuple[dict[tuple[int, int], list[tuple[int]]], float]:
        """method to find all paths between all nodes"""
        start = time.time()
        all_paths = {}
        nodes_pair = list(combinations(self.graph.nodes, 2))
        for source, target in nodes_pair:
            one_pair_paths, _ = self.find_path_pair(source, target)
            one_pair_paths_reverse = [list(reversed(path)) for path in one_pair_paths]
            all_paths[(source, target)] = one_pair_paths
            all_paths[(target, source)] = one_pair_paths_reverse
        elapsed_time = time.time() - start
        
        return all_paths, elapsed_time
