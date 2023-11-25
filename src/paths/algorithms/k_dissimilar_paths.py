from __future__ import annotations

from typing import Tuple, List

import networkx as nx
from gurobipy import Model, GRB, quicksum
from itertools import combinations
from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.utils.graph import path_similarity, calc_path_length


class KDissimilarPaths(BasePathAlgorithm):
    def find_path_pair(
        self, 
        source: int, 
        target: int
        ) -> Tuple[List[Tuple[int]], float]:
        """
        k-Dissimilar PathsをILPで解く関数
        """
        model = Model('k_dissimilar_paths')
        model.Params.OutputFlag = 0
        paths = list(nx.all_simple_paths(
            self.graph, source=source, target=target
            ))
        paths = [path for path in paths if calc_path_length(self.graph, path) <= self.length_limit]
        path_pairs = list(combinations(range(len(paths)), 2))

        # 変数の定義
        x = model.addVars(len(paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")
        theta = model.addVar(vtype=GRB.CONTINUOUS, name="theta")

        # 目的関数
        model.setObjectiveN(theta, 0, 1)
        model.setObjectiveN(
            quicksum(calc_path_length(self.graph, paths[i]) * x[i] for i in range(len(paths))), 1, 0
            )

        # 制約条件
        model.addConstr(
            quicksum(x[i] for i in range(len(paths))) == min(self.path_nums, len(paths)), "k_paths"
            )

        # y_ij の表現
        for i, j in path_pairs:
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j))

        # 選んだパス集合の類似度の合計です。
        model.addConstr(
            theta == quicksum(path_similarity(self.graph, paths[i], paths[j]) * y[i, j] for i, j in path_pairs),
            "theta_definition"
        )

        # 求解
        model.optimize()

        # パス集合の取得
        k_paths = [paths[i] for i in range(len(paths)) if x[i].X > 0.5]
        
        similarity_sum = theta.X

        return k_paths, similarity_sum
