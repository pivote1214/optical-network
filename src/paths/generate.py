from __future__ import annotations

from typing import Dict, Tuple, List, Any

import pickle
import gurobipy as gp
import tqdm
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_balanced_paths import KBalancedPath
from src.utils.graph import create_network
from src.utils.paths import RESULT_DIR


def generate_all_paths(
    network_name: str, 
    algorithm_name: str, 
    path_nums: int, 
    length_limit: int = 6300, 
    alpha: float = None
    ) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], List[Tuple[int]]]]:
    global graph
    
    if algorithm_name == "kSP":
        algorithm = KShortestPaths(graph, path_nums, length_limit)
        all_paths, elapsed_time = algorithm.find_all_paths()
    elif algorithm_name == "kDP":
        algorithm = KDissimilarPaths(graph, path_nums, length_limit)
        all_paths, elapsed_time = algorithm.find_all_paths()
    elif algorithm_name == "kBP":
        algorithm = KBalancedPath(graph, path_nums, length_limit)
        all_paths, elapsed_time = algorithm.find_all_paths(alpha)
    else:
        raise ValueError("Invalid algorithm name.")

    basic_info = {
        "network_name": network_name, 
        "algorithm": algorithm_name, 
        "k": path_nums, 
        "length_limit": length_limit, 
        "alpha": alpha, 
        "elapsed_time": elapsed_time
    }

    return basic_info, all_paths
        

def save_paths(
    basic_info: Dict[str, Any], 
    all_paths: Dict[Tuple[int, int], List[Tuple[int]]]
    ) -> None:
    """
    パスとアルゴリズムに関する情報をpickleファイルとして保存する関数
    """
    algorithm_name = basic_info['algorithm']
    network_name = basic_info['network_name']
    path_nums = basic_info['k']
    alpha = basic_info['alpha']
    
    file_name = f"{algorithm_name}_{network_name}_k={path_nums}_alpha={alpha}.pickle"
    full_path = RESULT_DIR / 'paths' / file_name
    content = {'basic_info': basic_info, 'all_paths': all_paths}
    with open(full_path, 'wb') as f:
        pickle.dump(content, f)


if __name__ == "__main__":
    # ダミー
    graph = gp.Model()
    # グラフの定義
    network_name = "NSF"
    graph = create_network(network_name)

    # パラメータの定義
    path_nums_list = [i for i in range(1, 6)]
    algorithm_list = ["kSP", "kDP", "kBP"]
    alpha_list = [round(0.1 * i, 2) for i in range(1, 10)]

    for path_nums in tqdm.tqdm(path_nums_list):
        for algorithm_name in tqdm.tqdm(algorithm_list, leave=False):
            if algorithm_name == "kBP":
                for alpha in tqdm.tqdm(alpha_list, leave=False):
                    basic_info, all_paths = generate_all_paths(
                        network_name, algorithm_name, path_nums, alpha=alpha
                        )
                    save_paths(basic_info, all_paths)
            else:
                basic_info, all_paths = generate_all_paths(
                    network_name, algorithm_name, path_nums
                    )
                save_paths(basic_info, all_paths)
