import time
import networkx as nx
from itertools import combinations
from gurobipy import Model, GRB, quicksum

from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.utils.graph import calc_path_length, calc_path_similarity


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
        ) -> tuple[list[tuple[int]], None]:
        """method to find path pair with k balanced paths algorithm"""
        model = Model('k_balanced_paths')
        model.Params.OutputFlag = 0
        paths = list(nx.all_simple_paths(
            self.graph, source=source, target=target
            ))
        paths = [path for path in paths if calc_path_length(self.graph, path) <= self.length_limit]
        path_pairs = list(combinations(range(len(paths)), 2))

        # variables
        x = model.addVars(len(paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")

        # objective function
        model.setObjective(
            quicksum(calc_path_length(self.graph, paths[i]) * x[i] for i in range(len(paths))), 
            GRB.MINIMIZE
            )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(paths))) == min(self.path_nums, len(paths)), 
            "k_paths"
            )

        for i, j in path_pairs:
            if i > j:
                continue
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j))

        model.addConstr(
            quicksum(
                calc_path_similarity(self.graph, paths[i], paths[j]) * y[i, j] 
                for i, j in path_pairs if i < j
                ) <= alpha * theta_min + (1 - alpha) * theta_max, 
            "theta_definition"
        )

        # optimize
        model.optimize()

        # get result
        k_paths = [paths[i] for i in range(len(paths)) if x[i].X > 0.5]

        return k_paths, None

    def find_all_paths(
        self, 
        alpha: float
        ) -> tuple[dict[tuple[int, int], list[tuple[int]]], float]:
        """method to find all paths between all nodes with k balanced paths algorithm"""
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

    def _calc_theta(self) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """method to calculate theta_min and theta_max for all nodes pair"""
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
            
            
