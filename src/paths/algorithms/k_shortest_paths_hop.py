import networkx as nx

from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.utils.graph import calc_path_length, calc_path_similarity


class KShortestPathsHop(BasePathAlgorithm):
    def find_path_pair(
        self, 
        source: int, 
        target: int
        ) -> tuple[list[tuple[int]], float]:
        """method to find path pair with k shortest paths algorithm"""
        simple_paths = nx.all_simple_paths(
            self.graph, source=source, target=target
        )
        simple_paths = [path for path in simple_paths 
                        if calc_path_length(self.graph, path) <= self.length_limit]
        k_paths = sorted(simple_paths, 
                         key=lambda path: (len(path), calc_path_length(self.graph, path)))[:self.path_nums]

        similarity_sum = sum(
            calc_path_similarity(self.graph, k_paths[i], k_paths[j])
            for i in range(len(k_paths))
            for j in range(i + 1, len(k_paths))
        )

        return k_paths, similarity_sum
