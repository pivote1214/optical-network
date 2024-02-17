import os
import time
import tqdm
import pickle
import pandas as pd
import gurobipy as gp

from src.utils.paths import PATHS_DIR, RESULT_DIR
from src.utils.graph import load_network
from src.paths.algorithms.repeat_dijkstra import repeat_dijkstra
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.paths.algorithms.k_shortest_paths_hop import KShortestPathsHop
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_balanced_paths import KBalancedPath


def generate_all_paths(
    graph_name: str, 
    algorithm_name: str, 
    path_nums: int, 
    length_limit: int = 6300, 
    alpha: float = None
    ) -> dict[tuple[int, int], list[tuple[int]]]:
    """generate all paths with given algorithm"""
    graph = load_network(graph_name)

    if algorithm_name == "kSP":
        algorithm = KShortestPaths(graph, path_nums, length_limit)
        all_paths, _ = algorithm.find_all_paths()
    elif algorithm_name == "kSP-hop":
        algorithm = KShortestPathsHop(graph, path_nums, length_limit)
        all_paths, _ = algorithm.find_all_paths()
    elif algorithm_name == "kDP":
        algorithm = KDissimilarPaths(graph, path_nums, length_limit)
        all_paths, _ = algorithm.find_all_paths()
    elif algorithm_name == "kSPwLO":
        algorithm = KBalancedPath(graph, path_nums, length_limit)
        all_paths, _ = algorithm.find_all_paths(alpha)
    else:
        raise ValueError("Algorithm name should be kSP, kSP-hop, kDP or kSPwLO.")

    return all_paths

def save_all_paths(file_dir: str, file_name: str, all_paths: dict[tuple[int, int], list[tuple[int]]]):
    """save paths by pickle"""
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_dir / file_name, "wb") as f:
        pickle.dump(all_paths, f)


if __name__ == "__main__":
    # dummy
    graph = gp.Model()

    # parameter
    graph_names = ["NSF", "EURO16", "JPN12"]
    path_nums_values = [i for i in range(2, 4)]
    algorithm_names = ["kSP", "kSP-hop", "kDP", "kSPwLO"]
    alpha_values = [round(0.1 * i, 2) for i in range(1, 10)]

    # generate all paths
    for graph_name in graph_names:
        print(f"Generating all paths for {graph_name}...")
        graph = load_network(graph_name)
        calc_time = pd.DataFrame(index=path_nums_values)
        for path_nums in tqdm.tqdm(path_nums_values):
            for algorithm_name in tqdm.tqdm(algorithm_names, leave=False):
                file_dir = PATHS_DIR / graph_name / algorithm_name
                if algorithm_name == "kSPwLO":
                    for alpha in tqdm.tqdm(alpha_values, leave=False):
                        start = time.time()
                        all_pahts = generate_all_paths(
                            graph_name, algorithm_name, path_nums, alpha=alpha
                            )
                        end = time.time()
                        calc_time.loc[path_nums, f"{algorithm_name}_{alpha}"] = end - start
                        file_name = f"k={path_nums}_alpha={alpha}.pickle"
                        save_all_paths(file_dir, file_name, all_pahts)
                else:
                    start = time.time()
                    all_pahts = generate_all_paths(
                        graph_name, algorithm_name, path_nums
                        )
                    end = time.time()
                    calc_time.loc[path_nums, algorithm_name] = end - start
                    file_name = f"k={path_nums}.pickle"
                    save_all_paths(file_dir, file_name, all_pahts)
        calc_time.to_csv(RESULT_DIR / "paths"/ f"{graph_name}_calc_time.csv")

    print("All paths are generated.")
