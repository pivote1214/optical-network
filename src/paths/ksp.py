from typing import Any

import networkx as nx

from src.graph import calc_all_simple_paths, calc_path_weight
from src.paths._registry import register_path_selector
from src.paths.base import BasePathSelectorSinglePair

__all__ = ["ksp"]


class KShortestPaths(BasePathSelectorSinglePair):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int,
        length_metric: str = "hop",
    ):
        super().__init__(graph, n_paths, max_length, length_metric=length_metric)
        self.length_metric = length_metric

    def select_paths_single_pair(
        self,
        source: Any,
        target: Any,
    ) -> list[tuple[Any]]:
        """method to find path pair with k shortest paths algorithm"""
        all_simple_paths = self.all_simple_paths[(source, target)]
        # パスの重み順にソート
        if self.length_metric == "physical-length":
            all_simple_paths.sort(
                key=lambda path: (
                    calc_path_weight(self.graph, path),
                    calc_path_weight(self.graph, path, "hop"),
                )
            )
        elif self.length_metric == "hop":
            all_simple_paths.sort(
                key=lambda path: (
                    calc_path_weight(self.graph, path, "hop"),
                    calc_path_weight(self.graph, path),
                    path,
                )
            )
        else:
            raise ValueError('path_weight must be "physical-length" or "hop"')
        # 重みの小さなk本のパスを取得
        k_paths = all_simple_paths[:self.n_paths_per_pair[source, target]]

        return k_paths


@register_path_selector
def ksp(
    graph: nx.DiGraph, n_paths: int, max_length: int, **kwargs: Any
) -> KShortestPaths:
    return KShortestPaths(graph, n_paths, max_length, **kwargs)
