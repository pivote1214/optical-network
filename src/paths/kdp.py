from itertools import combinations
from typing import Any

import networkx as nx
from gurobipy import GRB, Model, quicksum

from src.graph import calc_all_simple_paths, calc_path_similarity, calc_path_weight
from src.paths._registry import register_path_selector
from src.paths.base import BasePathSelectorSinglePair

__all__ = ["kdp"]


class KDissimilarPaths(BasePathSelectorSinglePair):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int = 6300,
        sim_metric: str = "physical-length",
    ):
        super().__init__(graph, n_paths, max_length, sim_metric=sim_metric)
        self.sim_metric = sim_metric

    def select_paths_single_pair(self, source: Any, target: Any) -> list[tuple[Any]]:
        """
        method to find path pair with k dissimilar paths algorithm
        
        Parameters
        ----------
        source: Any
            source node
        target: Any
            target node
        
        Returns
        -------
        list[tuple[Any]]
            list of selected paths
        """
        model = Model(f"k_dissimilar_paths_{source}-{target}")
        model.setParam("LogToConsole", 0)
        model.setParam(GRB.Param.Threads, 6)

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
        theta = model.addVar(vtype=GRB.CONTINUOUS, name="theta")

        # objective function
        model.setObjectiveN(theta, 0, 1)
        model.setObjectiveN(
            quicksum(
                calc_path_weight(self.graph, all_simple_paths[i]) * x[i]
                for i in range(len(all_simple_paths))
            ),
            1,
            0,
        )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(all_simple_paths)))
            == min(self.n_paths_per_pair[source, target], len(all_simple_paths)),
            "k_paths",
        )

        # 最短パスを必ず選択
        model.addConstr(x[shortest_path_idx] == 1, "shortest_path")

        for p_i, p_j in path_pairs:
            if p_i > p_j:
                continue
            model.addConstr(y[p_i, p_j] <= x[p_i], "y_constr_{}_{}".format(p_i, p_j))
            model.addConstr(y[p_i, p_j] <= x[p_j], "y_constr_{}_{}".format(p_j, p_i))
            model.addConstr(
                y[p_i, p_j] >= x[p_i] + x[p_j] - 1, "y_constr_sum_{}_{}".format(p_i, p_j)
            )

        model.addConstr(
            theta
            == quicksum(
                calc_path_similarity(
                    self.graph,
                    all_simple_paths[i],
                    all_simple_paths[j],
                    metric=self.sim_metric,
                )
                * y[i, j]
                for i, j in path_pairs
                if i < j
            ),
            "theta_definition",
        )

        # optimize
        model.optimize()
        # get results
        k_paths = [
            all_simple_paths[i] for i in range(len(all_simple_paths)) if x[i].X > 0.5
        ]

        return k_paths


@register_path_selector
def kdp(
    graph: nx.DiGraph, n_paths: int, max_length: int = 6300, **kwargs: Any
) -> KDissimilarPaths:
    return KDissimilarPaths(graph, n_paths, max_length, **kwargs)
