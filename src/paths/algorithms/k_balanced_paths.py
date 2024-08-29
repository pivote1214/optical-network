import os
from itertools import combinations

from gurobipy import Model, GRB, quicksum

from utils.network import calc_path_weight, calc_path_similarity
from utils.files import save_pickle
from utils.namespaces import PATHS_DIR
from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_shortest_paths import KShortestPaths


class KBalancedPaths(PathSelectionAlgorithm):
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: dict = {'path_weight': 'physical-length', 'sim_weight': 'physical-length', 'alpha': 0.3}, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, params, length_limit)
        self.all_theta_min, self.all_theta_max = self._calc_theta()
    
    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ) -> list[tuple[int]]:
        """method to find path pair with k balanced paths algorithm"""
        # initialize
        alpha = self.params['alpha']
        theta_min = self.all_theta_min[(source, target)]
        theta_max = self.all_theta_max[(source, target)]
        # create model
        model = Model('k_balanced_paths')
        model.Params.OutputFlag = 0
        all_simple_paths = self._calc_all_simple_paths(source, target)
        path_pairs = list(combinations(range(len(all_simple_paths)), 2))

        # variables
        x = model.addVars(len(all_simple_paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")

        # objective function
        model.setObjective(
            quicksum(
                calc_path_weight(self.graph, all_simple_paths[i], metrics=self.params['path_weight']) * x[i] 
                for i in range(len(all_simple_paths))
                ), 
            GRB.MINIMIZE
            )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(all_simple_paths))) == min(self.n_paths, len(all_simple_paths)), 
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
                calc_path_similarity(self.graph, all_simple_paths[i], all_simple_paths[j], edge_weight=self.params['sim_weight']) * y[i, j] 
                for i, j in path_pairs if i < j
                ) <= alpha * theta_min + (1 - alpha) * theta_max, 
            "theta_definition"
        )

        # optimize
        model.optimize()
        # get result
        k_paths = [all_simple_paths[i] for i in range(len(all_simple_paths)) if x[i].X > 0.5]

        return k_paths

    def _calc_theta(self) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """method to calculate theta_min and theta_max for all nodes pair"""
        # 初期化
        all_theta_min = {}
        all_theta_max = {}

        ksp_alogrithm = KShortestPaths(self.graph_name, self.n_paths, self.params, self.length_limit)
        kdp_algorithm = KDissimilarPaths(self.graph_name, self.n_paths, self.params, self.length_limit)

        nodes_pair = list(combinations(self.graph.nodes(), 2))
        for source, target in nodes_pair:
            k_shortest_paths = ksp_alogrithm.select_k_paths_single_pair(source, target)
            k_dissimilar_paths = kdp_algorithm.select_k_paths_single_pair(source, target)
            theta_min = self._calc_total_similarity(k_shortest_paths)
            theta_max = self._calc_total_similarity(k_dissimilar_paths)
            all_theta_min[(source, target)] = theta_min
            all_theta_min[(target, source)] = theta_min
            all_theta_max[(source, target)] = theta_max
            all_theta_max[(target, source)] = theta_max

        return all_theta_min, all_theta_max

    def save_selected_paths_all_pairs(self) -> None:
        all_paths = self.select_k_paths_all_pairs()
        output_file = os.path.join(
            PATHS_DIR, 
            'k-shortest-paths-with-similarity-constraint', 
            self.graph_name, 
            f'path_weight_{self.params["path_weight"]}', 
            f'sim_weight_{self.params["sim_weight"]}', 
            f'alpha_{self.params["alpha"]}'.replace(".", "d"), 
            f'n-paths_{self.n_paths}.pkl'
            )
        save_pickle(all_paths, output_file)
