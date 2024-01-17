import networkx as nx
from itertools import combinations
from gurobipy import Model, GRB, quicksum

from src.paths.algorithms.base_algorithm import BasePathAlgorithm
from src.utils.graph import calc_path_similarity, calc_path_length


class KDissimilarPaths(BasePathAlgorithm):
    def find_path_pair(
        self, 
        source: int, 
        target: int
        ) -> tuple[list[tuple[int]], float]:
        """method to find path pair with k dissimilar paths algorithm"""
        model = Model('k_dissimilar_paths')
        model.Params.OutputFlag = 0
        paths = list(nx.all_simple_paths(
            self.graph, source=source, target=target
            ))
        paths = [path for path in paths if calc_path_length(self.graph, path) <= self.length_limit]
        path_pairs = list(combinations(range(len(paths)), 2))

        # variables
        x = model.addVars(len(paths), vtype=GRB.BINARY, name="x")
        y = model.addVars(path_pairs, vtype=GRB.BINARY, name="y")
        theta = model.addVar(vtype=GRB.CONTINUOUS, name="theta")

        # objective function
        model.setObjectiveN(theta, 0, 1)
        model.setObjectiveN(
            quicksum(calc_path_length(self.graph, paths[i]) * x[i] for i in range(len(paths))), 1, 0
            )

        # constraint
        model.addConstr(
            quicksum(x[i] for i in range(len(paths))) == min(self.path_nums, len(paths)), "k_paths"
            )

        for i, j in path_pairs:
            if i > j:
                continue
            model.addConstr(y[i, j] <= x[i], "y_constr_{}_{}".format(i, j))
            model.addConstr(y[i, j] <= x[j], "y_constr_{}_{}".format(j, i))
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, "y_constr_sum_{}_{}".format(i, j))

        model.addConstr(
            theta == quicksum(calc_path_similarity(self.graph, paths[i], paths[j]) * y[i, j] 
                              for i, j in path_pairs if i < j),
            "theta_definition"
        )

        # optimize
        model.optimize()

        # get results
        k_paths = [paths[i] for i in range(len(paths)) if x[i].X > 0.5]
        
        similarity_sum = theta.X

        return k_paths, similarity_sum
