from __future__ import annotations

from typing import Dict, Tuple, List

import time
import networkx as nx
from gurobipy import Model, GRB, quicksum
from itertools import combinations
from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.utils.graph import path_length, path_similarity


class KBalancedPath(BasePathAlgorithm):
    def __init__(
        self, 
        graph: nx.Graph, 
        path_nums: int, 
        length_limit: int = 6300
        ):
        super().__init__(graph, path_nums, length_limit)
        self.all_theta_min, self.all_theta_max, self.preprocess_time = self._calc_theta()
    
    def find_path_pair(
        self, 
        source: int, 
        target: int, 
        alpha: float, 
        theta_min: float, 
        theta_max: float
        ) -> Tuple[List[Tuple[int]], None]:
        """
        k-Balanced PathをILPで解く関数
        """
        # _, theta_min = KShortestPaths(
        #     self.graph, self.path_nums, self.length_limit
        #     ).find_path_pair(source, target)
        # _, theta_max = KDissimilarPaths(
        #     self.graph, self.path_nums, self.length_limit
        #     ).find_path_pair(source, target)

        model = Model('k_balanced_paths')
        model.Params.OutputFlag = 0
        paths = list(nx.all_simple_paths(
            self.graph, source=source, target=target, cutoff=self.length_limit
            ))
        path_pairs = list(combinations(range(len(paths)), 2))

        # 変数の定義
        x = model.addVars(len(paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")

        # 目的関数
        model.setObjective(
            quicksum(path_length(self.graph, paths[i]) * x[i] for i in range(len(paths))), 
            GRB.MINIMIZE
            )

        # 制約条件
        model.addConstr(
            quicksum(x[i] for i in range(len(paths))) == min(self.path_nums, len(paths)), 
            "k_paths"
            )

        # y_ij の表現
        for i, j in path_pairs:
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j))

        # 類似度の制約
        model.addConstr(
            quicksum(
                path_similarity(self.graph, paths[i], paths[j]) * y[i, j] for i, j in path_pairs
                ) <= alpha * theta_min + (1 - alpha) * theta_max, 
            "theta_definition"
        )

        # 求解
        model.optimize()

        # パス集合の取得
        k_paths = [paths[i] for i in range(len(paths)) if x[i].X > 0.5]

        return k_paths, None

    def find_all_paths(
        self, 
        alpha: float
        ) -> Tuple[List[List[Tuple[int]]], float]:
        """
        kBPですべての頂点対間のk個のパスを求めるメソッド
        """
        start = time.time()
        all_paths = {}
        nodes_pair = list(combinations(self.graph.nodes(), 2))
        for source, target in nodes_pair:
            theta_min = self.all_theta_min[(source, target)]
            theta_max = self.all_theta_max[(source, target)]
            one_pair_paths, _ = self.find_path_pair(source, target, alpha, theta_min, theta_max)
            one_pair_paths_reversed = [list(reversed(path)) for path in one_pair_paths]
            all_paths[(source, target)] = one_pair_paths
            all_paths[(target, source)] = one_pair_paths_reversed
        elapsed_time = round(time.time() - start, 2) + self.preprocess_time

        return all_paths, elapsed_time

    def _calc_theta(self) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """
        すべての頂点対間のtheta_min, theta_maxを求めるメソッド
        """
        all_theta_min = {}
        all_theta_max = {}

        start = time.time()
        k_shortest_paths = KShortestPaths(self.graph, self.path_nums)
        k_dissimilar_paths = KDissimilarPaths(self.graph, self.path_nums)

        nodes_pair = list(combinations(self.graph.nodes(), 2))
        for source, target in nodes_pair:
            _, theta_min = k_dissimilar_paths.find_path_pair(source, target)
            _, theta_max = k_shortest_paths.find_path_pair(source, target)
            all_theta_min[(source, target)] = theta_min
            all_theta_min[(target, source)] = theta_min
            all_theta_max[(source, target)] = theta_max
            all_theta_max[(target, source)] = theta_max

        preprocess_time = round(time.time() - start, 2)

        return all_theta_min, all_theta_max, preprocess_time
            
            
