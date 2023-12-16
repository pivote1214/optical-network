from __future__ import annotations

from typing import Dict, Tuple, List, Any

import pickle
import gurobipy as gp
import tqdm
from src.utils.graph import load_network
from src.utils.paths import PATHS_DIR
from src.paths.calc_metrics import calc_all_metrics
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_balanced_paths import KBalancedPath


def generate_all_paths(
    network_name: str, 
    algorithm_name: str, 
    path_nums: int, 
    length_limit: int = 6300, 
    alpha: float = None
    ) -> Dict[str, Any]:
    """generate all paths with given algorithm"""
    graph = load_network(network_name)

    if algorithm_name == "kSP":
        algorithm = KShortestPaths(graph, path_nums, length_limit)
        paths, elapsed_time = algorithm.find_all_paths()
    elif algorithm_name == "kDP":
        algorithm = KDissimilarPaths(graph, path_nums, length_limit)
        paths, elapsed_time = algorithm.find_all_paths()
    elif algorithm_name == "kSPwLO":
        algorithm = KBalancedPath(graph, path_nums, length_limit)
        paths, elapsed_time = algorithm.find_all_paths(alpha)
    else:
        raise ValueError("Invalid algorithm name.")

    metcis = calc_all_metrics(graph, paths)
    metcis["elapsed_time"] = elapsed_time

    parameters = {
        "network_name": network_name, 
        "algorithm": algorithm_name, 
        "k": path_nums, 
        "length_limit": length_limit, 
        "alpha": alpha
    }

    paths_data = {"parameters": parameters, "paths": paths, "metrics": metcis}

    return paths_data
        

def save_paths(paths_data) -> None:
    """save parameters, paths and metrics as pickle file"""
    algorithm_name = paths_data['parameters']['algorithm']
    network_name = paths_data['parameters']['network_name']
    path_nums = paths_data['parameters']['k']
    alpha = paths_data['parameters']['alpha']
    
    file_name = f"{algorithm_name}_k={path_nums}_alpha={alpha}.pickle"
    full_path = PATHS_DIR /network_name / file_name
    with open(full_path, 'wb') as f:
        pickle.dump(paths_data, f)


if __name__ == "__main__":
    # ダミー
    graph = gp.Model()
    # グラフの定義
    network_name = "EURO16"
    graph = load_network(network_name)

    # パラメータの定義
    path_nums_list = [i for i in range(1, 6)]
    algorithm_list = ["kSP", "kDP", "kSPwLO"]
    alpha_list = [round(0.1 * i, 2) for i in range(1, 10)]

    for path_nums in tqdm.tqdm(path_nums_list):
        for algorithm_name in tqdm.tqdm(algorithm_list, leave=False):
            if algorithm_name == "kSPwLO":
                for alpha in tqdm.tqdm(alpha_list, leave=False):
                    paths_data = generate_all_paths(
                        network_name, algorithm_name, path_nums, alpha=alpha
                        )
                    save_paths(paths_data)
            else:
                paths_data = generate_all_paths(
                    network_name, algorithm_name, path_nums
                    )
                save_paths(paths_data)
