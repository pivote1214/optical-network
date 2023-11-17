from __future__ import annotations

from typing import Tuple, List

import networkx as nx
from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.utils.graph import calc_path_length, path_similarity


class KShortestPaths(BasePathAlgorithm):
    def find_path_pair(
        self, 
        source: int, 
        target: int
        ) -> Tuple[List[Tuple[int]], float]:
        """
        指定された2点対間のk-Shorest Pathsを求める関数
        """
        simple_paths = nx.all_simple_paths(
            self.graph, source=source, target=target, cutoff=self.length_limit
        )
        k_paths = sorted(simple_paths, key=lambda p: calc_path_length(self.graph, p))[:self.path_nums]

        similarity_sum = sum(
            path_similarity(self.graph, k_paths[i], k_paths[j])
            for i in range(len(k_paths))
            for j in range(i + 1, len(k_paths))
        )

        return k_paths, similarity_sum
