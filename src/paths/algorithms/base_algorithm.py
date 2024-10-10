import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
    )

from itertools import combinations

import networkx as nx

from utils.files import save_pickle, set_paths_file_path
from utils.network import calc_path_similarity, calc_path_weight, load_network


class PathSelectionAlgorithm:
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        length_limit: int = 6300, 
        ) -> None:
        self.graph_name = graph_name
        self.graph = load_network(graph_name)
        self.n_paths = n_paths
        self.length_limit = length_limit
        self.params = None

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ):
        """method to select k paths for single pair"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def select_k_paths_all_pairs(self) -> dict[tuple[int, int], list[tuple[int]]]:
        """method to select k paths for all pairs"""
        all_paths = {}
        nodes_pair = list(combinations(self.graph.nodes, 2))
        for source, target in nodes_pair:
            one_pair_paths = self.select_k_paths_single_pair(source, target)
            one_pair_paths_reverse = [list(reversed(path)) for path in one_pair_paths]
            all_paths[(source, target)] = one_pair_paths
            all_paths[(target, source)] = one_pair_paths_reverse
        
        return all_paths

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
            total_similarity += calc_path_similarity(self.graph, k_paths[i], k_paths[j], edge_weight=self.params.sim_metric)

        return total_similarity

    def save_selected_paths(self) -> None:
        """method to save selected paths"""
        # set file path
        file_paths = set_paths_file_path(
            algorithm=self.__class__.__name__, 
            network_name=self.graph_name, 
            params=self.params, 
            n_paths=self.n_paths
            )
        # if folder exists, continue
        if os.path.exists(file_paths):
            return
        else:
            os.makedirs(os.path.dirname(file_paths), exist_ok=True)
        # select paths
        candidate_paths_set = self.select_k_paths_all_pairs()
        # save paths
        save_pickle(candidate_paths_set, file_paths)
