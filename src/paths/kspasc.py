from itertools import combinations
from typing import Any

import networkx as nx
from gurobipy import GRB, Model, quicksum

from src.graph import (
    calc_all_simple_paths,
    calc_path_similarity,
    calc_path_weight,
    calc_total_similarity,
)
from src.paths._factory import create_path_selector
from src.paths._registry import register_path_selector
from src.paths.base import BasePathSelectorSinglePair

__all__ = ["kspasc"]


class KSPAlphaSimilarityConstraint(BasePathSelectorSinglePair):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int = 6300,
        length_metric: str = "hop",
        sim_metric: str = "physical-length",
        alpha: float = 0.5,
    ):
        super().__init__(
            graph,
            n_paths,
            max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            alpha=alpha,
        )
        self.length_metric = length_metric
        self.sim_metric = sim_metric
        self.alpha = alpha
        if not self.is_calculated:
            self.all_theta_min = self._calc_theta_min()
            self.all_theta_max = self._calc_theta_max()

    def select_paths_single_pair(self, source: Any, target: Any) -> list[tuple[Any]]:
        """method to find path pair with k balanced paths algorithm"""
        # prepare
        theta_min = self.all_theta_min[(source, target)]
        theta_max = self.all_theta_max[(source, target)]
        # create model
        model = Model("kspasc")
        model.setParam("LogToConsole", 0)
        model.setParam(GRB.Param.Threads, 6)
        # calc all simple paths
        all_simple_paths = self.all_simple_paths[(source, target)]
        path_pairs = list(combinations(range(len(all_simple_paths)), 2))
        
        # 最短パスを特定
        shortest_path = min(
            all_simple_paths,
            key=lambda path: calc_path_weight(self.graph, path)
        )
        shortest_path_idx = all_simple_paths.index(shortest_path)

        # variables
        x = model.addVars(len(all_simple_paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")

        # objective function
        model.setObjective(
            quicksum(
                calc_path_weight(
                    self.graph, all_simple_paths[i], metrics=self.length_metric
                )
                * x[i]
                for i in range(len(all_simple_paths))
            ),
            GRB.MINIMIZE,
        )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(all_simple_paths)))
            == min(self.n_paths_per_pair[source, target], len(all_simple_paths)),
            "k_paths",
        )

        # 最短パスを必ず選択
        model.addConstr(x[shortest_path_idx] == 1, "shortest_path")

        for i, j in path_pairs:
            if i > j:
                continue
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(
                y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j)
            )

        model.addConstr(
            quicksum(
                calc_path_similarity(
                    self.graph,
                    all_simple_paths[i],
                    all_simple_paths[j],
                    metric=self.sim_metric,
                )
                * y[i, j]
                for i, j in path_pairs
                if i < j
            )
            <= self.alpha * theta_min + (1 - self.alpha) * theta_max,
            "theta_definition",
        )

        # optimize
        model.optimize()
        # get result
        k_paths = [
            all_simple_paths[i] for i in range(len(all_simple_paths)) if x[i].X > 0.5
        ]

        return k_paths

    def _calc_theta_min(self) -> dict[tuple[Any, Any], float]:
        """method to calculate theta_min and theta_max for all nodes pair"""
        all_theta_min = {}
        # kDPの呼び出し
        kdp = create_path_selector(
            "kdp", self.graph, self.n_paths, self.max_length, sim_metric=self.sim_metric
        )
        kdp_paths = kdp._select_all_paths()
        for (source, target), paths in kdp_paths.items():
            theta_min = calc_total_similarity(self.graph, paths, self.sim_metric)
            all_theta_min[(source, target)] = theta_min

        return all_theta_min

    def _calc_theta_max(self) -> dict[tuple[Any, Any], float]:
        """method to calculate theta_max for all nodes pair"""
        all_theta_max = {}
        # kSPの呼び出し
        ksp = create_path_selector(
            "ksp",
            self.graph,
            self.n_paths,
            self.max_length,
            length_metric=self.length_metric,
        )
        ksp_paths = ksp._select_all_paths()
        for (source, target), paths in ksp_paths.items():
            theta_max = calc_total_similarity(self.graph, paths, self.sim_metric)
            all_theta_max[(source, target)] = theta_max

        return all_theta_max


@register_path_selector
def kspasc(
    graph: nx.DiGraph, n_paths: int, max_length: int = 6300, **kwargs
) -> KSPAlphaSimilarityConstraint:
    return KSPAlphaSimilarityConstraint(graph, n_paths, max_length, **kwargs)
