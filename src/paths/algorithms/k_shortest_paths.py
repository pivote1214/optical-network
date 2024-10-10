import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
    )

from dataclasses import dataclass

from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from utils.network import calc_path_weight


@dataclass(frozen=True)
class KShortestPathsParams:
    length_metric:  str


class KShortestPaths(PathSelectionAlgorithm):
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: KShortestPathsParams, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, length_limit)
        self.params = params

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ) -> tuple[list[tuple[int]], float]:
        """method to find path pair with k shortest paths algorithm"""
        all_simple_paths = self._calc_all_simple_paths(source, target)
        # パスの重み順にソート
        if self.params.length_metric == 'physical-length':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path), 
                                                    calc_path_weight(self.graph, path, 'hop')))
            # print(all_simple_paths)
        elif self.params.length_metric == 'hop':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path, 'hop'), 
                                                    calc_path_weight(self.graph, path), 
                                                    path))
            # print(all_simple_paths)
        elif self.params.length_metric == 'expected-used-slots':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path, 'expected-used-slots'), 
                                                    calc_path_weight(self.graph, path, 'hop'), 
                                                    calc_path_weight(self.graph, path)))
        else:
            raise ValueError('path_weight must be "physical-length", "hop" or "expected-used-slots')
        # 重みの小さなk本のパスを取得
        k_paths = all_simple_paths[:self.n_paths]

        return k_paths
