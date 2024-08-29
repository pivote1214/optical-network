import os
from pprint import pprint
import sys

from networkx import path_weight
sys.path.append('../..')

import pickle

import numpy as np
import pandas as pd
import networkx as nx

from utils.namespaces import DATA_DIR, OUT_DIR, PATHS_DIR
from utils.network import calc_path_weight, calc_path_similarity, load_network


def calc_avg_path_len_hops(
    graph: nx.Graph,
    paths: dict[tuple[int, int], list[tuple[int]]]
    ) -> float:
    """
    パスの長さとホップ数の平均を計算する関数
    """
    length_sum = sum(
        sum([calc_path_weight(graph, path) for path in paths[key]]) 
        for key in paths.keys()
        )
    hop_sum = sum(
        sum([len(path) - 1 for path in paths[key]]) 
        for key in paths.keys()
        )
    path_num = sum([len(paths[key]) for key in paths.keys()])

    length_ave, hop_ave = length_sum / path_num, hop_sum / path_num

    return length_ave, hop_ave


def calc_avg_path_sim(
    graph: nx.Graph, 
    paths: dict[tuple[int, int], list[tuple[int]]]
    ) -> float:
    """
    パスの類似度の平均を計算する関数
    """
    similarity_sum = sum(
        sum(calc_path_similarity(graph, paths[key][i], paths[key][j]) 
            for i in range(len(paths[key]))
            for j in range(i + 1, len(paths[key]))) 
        for key in paths.keys()
        )
    path_pair_num = sum(
        len(paths[key]) * (len(paths[key]) - 1) // 2 
        for key in paths.keys()
        )

    # 点対のパスが1の場合
    if path_pair_num == 0:
        similarity_ave = 0
    else:
        similarity_ave = similarity_sum / path_pair_num

    return similarity_ave


def calc_avg_path_nums(
    paths: dict[tuple[int, int], list[tuple[int]]]
    ) -> float:
    """
    パスの数の平均を計算する関数
    """
    path_num = sum([len(paths[key]) for key in paths.keys()])
    path_num_ave = path_num / len(paths.keys())

    return path_num_ave


def calc_edge_usage_metrics(
    graph: nx.Graph, 
    paths: dict[tuple[int, int], list[tuple[int]]]
    ) -> dict[tuple[int, int], int]:
    """
    辺の使用回数の辞書を計算する関数
    """
    edge_usage = {(min(u, v), max(u, v)): 0 for u, v in graph.edges()}

    # 各パスにおける辺の使用回数をカウント
    for path_list in paths.values():
        for path in path_list:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = (min(u, v), max(u, v))
                edge_usage[edge] += 1

    return edge_usage


def calc_all_metrics(
    graph: nx.Graph, 
    paths: dict[tuple[int, int], list[tuple[int]]]
    ) -> dict[str, any]:
    """calculate all metrics"""
    length_ave, hop_ave = calc_avg_path_len_hops(graph, paths)
    similarity_ave = calc_avg_path_sim(graph, paths)
    path_num_ave = calc_avg_path_nums(paths)
    edge_usage = calc_edge_usage_metrics(graph, paths)
    edge_usage_ave = np.mean(list(edge_usage.values()))
    edge_usage_std = np.std(list(edge_usage.values()))

    metrics = {
        'length_ave': length_ave,
        'hop_ave': hop_ave,
        'similarity_ave': similarity_ave,
        'path_num_ave': path_num_ave, 
        'edge_usage': edge_usage, 
        'edge_usage_ave': edge_usage_ave, 
        'edge_usage_std': edge_usage_std
    }

    return metrics

if __name__ == '__main__':
    def get_path_name(
        algorithm: str, 
        graph: str, 
        path_weight: str, 
        sim_weight: str, 
        alpha: str, 
        path_num: int
        ) -> str:
        file_path = os.path.join(
            PATHS_DIR, 
            algorithm, 
            graph, 
            f'path_weight_{path_weight}', 
            f'sim_weight_{sim_weight}', 
            f'alpha_{alpha}', 
            f'n-paths_{path_num}.pkl'
            )
        return file_path

    # parameters
    algorithm = 'k-shortest-paths-with-similarity-constraint'
    # network_names = ['NSF', 'JPN12']
    network_names = ['NSF']
    path_weights = ['hop', 'expected-used-slots']
    sim_weights = ['all-one', 'physical-length']
    alpha_values = ['0d0', '0d25', '0d5', '0d75', '1d0']
    # n_paths = [2, 3]
    n_paths = [3]
    metrics = ['length_ave', 'hop_ave', 'similarity_ave']
    # initialize metrics_table
    for network_name in network_names:
        graph = load_network(network_name)
        for p_weight in path_weights:
            for s_weight in sim_weights:
                for alpha in alpha_values:
                    for k in n_paths:
                        file_path = get_path_name(
                            algorithm, network_name, p_weight, s_weight, alpha, k
                            )
                        with open(file_path, 'rb') as f:
                            all_paths = pickle.load(f)
                        all_metrics = calc_all_metrics(graph, all_paths)
                        print(f"network: {network_name}, path_weight: {p_weight}, sim_weight: {s_weight}, alpha: {alpha}, k: {k}")
                        pprint(all_metrics)
                        print()
                            
        # for k in range(2, 4):
        #     for algo_name in algo_names:
        #         if algo_name == 'kSPwLO':
        #             for alpha in np.arange(0.1, 1.0, 0.1):
        #                 file_path = PATHS_DIR / network_name / algo_name / f'k={k}_alpha={round(alpha, 2)}.pkl'
        #                 with open(PATHS_DIR/ 'NSF' / file_path, 'rb') as f:
        #                     all_paths = pickle.load(f)
        #                 all_metrics = calc_all_metrics(graph, all_paths)
        #                 for metric in metrics:
        #                     metrics_table.loc[f"k={k}_{algo_name}_{round(alpha, 2)}", metric] = all_metrics[metric]
        #         else:
        #             file_path = PATHS_DIR / network_name / algo_name / f'k={k}.pkl'
        #             with open(PATHS_DIR / 'NSF' / file_path, 'rb') as f:
        #                 all_paths = pickle.load(f)
        #             all_metrics = calc_all_metrics(graph, all_paths)
        #             for metric in metrics:
        #                 metrics_table.loc[f"k={k}_{algo_name}", metric] = all_metrics[metric]

        # full_table_path = OUT_DIR / 'paths' / f'{network_name}_metrics_table.csv'
        # metrics_table.to_csv(full_table_path)
