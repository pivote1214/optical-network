import os
from itertools import combinations

from gurobipy import GRB, Model, quicksum

from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from utils.files import save_pickle, set_paths_file_path
from utils.namespaces import PATHS_DIR
from utils.network import calc_path_similarity, calc_path_weight


class KDissimilarPaths(PathSelectionAlgorithm):
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: dict = {'sim_weight': 'physical-length'}, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, params, length_limit)

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ) -> list[tuple[int]]:
        """method to find path pair with k dissimilar paths algorithm"""
        model = Model('k_dissimilar_paths')
        model.Params.OutputFlag = 0
        all_simple_paths = self._calc_all_simple_paths(source, target)
        path_pairs = list(combinations(range(len(all_simple_paths)), 2))

        # variables
        x = model.addVars(len(all_simple_paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")
        theta = model.addVar(vtype=GRB.CONTINUOUS, name="theta")

        # objective function
        model.setObjectiveN(theta, 0, 1)
        model.setObjectiveN(
            quicksum(calc_path_weight(self.graph, all_simple_paths[i]) * x[i] for i in range(len(all_simple_paths))), 1, 0
            )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(all_simple_paths))) == min(self.n_paths, len(all_simple_paths)), "k_paths"
            )

        for i, j in path_pairs:
            if i > j:
                continue
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j))

        model.addConstr(
            theta == \
                quicksum(
                    calc_path_similarity(
                        self.graph, 
                        all_simple_paths[i], 
                        all_simple_paths[j], 
                        edge_weight=self.params['sim_weight']
                        ) * y[i, j] 
                    for i, j in path_pairs if i < j
                    ),
            "theta_definition"
        )

        # optimize
        model.optimize()
        # get results
        k_paths = [all_simple_paths[i] for i in range(len(all_simple_paths)) if x[i].X > 0.5]

        return k_paths

    def save_selected_paths_all_pairs(self) -> None:
        all_paths = self.select_k_paths_all_pairs()
        output_file = set_paths_file_path(
            algorithm='k-dissimilar-paths', 
            network_name=self.graph_name, 
            params=self.params, 
            n_paths=self.n_paths
            )
        save_pickle(all_paths, output_file)
