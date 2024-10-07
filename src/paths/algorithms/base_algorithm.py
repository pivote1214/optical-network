from itertools import combinations
from typing import Any

import networkx as nx

from utils.network import calc_path_similarity, calc_path_weight, load_network


class PathSelectionAlgorithm:
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: Any, 
        length_limit: int = 6300, 
        ): # TODO: params -> None
        self.graph_name = graph_name
        self.graph = load_network(graph_name)
        self.n_paths = n_paths
        self.params = params
        self.length_limit = length_limit

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ):
        """method to select k paths for single pair"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def select_k_paths_all_pairs(self) -> dict[tuple[int, int]]:
        """method to select k paths for all pairs"""
        all_paths = {}
        nodes_pair = list(combinations(self.graph.nodes, 2))
        for source, target in nodes_pair:
            one_pair_paths = self.select_k_paths_single_pair(source, target)
            one_pair_paths_reverse = [list(reversed(path)) for path in one_pair_paths]
            all_paths[(source, target)] = one_pair_paths
            all_paths[(target, source)] = one_pair_paths_reverse
        
        return all_paths

    def save_selected_paths_all_pairs(self) -> None:
        """method to save selected paths for all pairs"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def _calc_all_simple_paths(
        self, 
        source: int, 
        target: int
        ) -> list[list[int]]:
        """method to calculate all simple paths between source and target"""
        all_simple_paths = list(nx.all_simple_paths(self.graph, source=source, target=target))
        all_simple_paths = [path for path in all_simple_paths if calc_path_weight(self.graph, path) <= self.length_limit]

        return all_simple_paths

    def _calc_total_similarity(self, k_paths) -> float:
        """method to calculate total similarity of k paths"""
        total_similarity = 0
        path_pairs = list(combinations(range(len(k_paths)), 2))
        for i, j in path_pairs:
            total_similarity += calc_path_similarity(self.graph, k_paths[i], k_paths[j], edge_weight=self.params['sim_weight'])

        return total_similarity
