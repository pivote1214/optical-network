from __future__ import annotations

from typing import Any, Dict, Tuple, List

import pickle
import numpy as np
import pandas as pd
import networkx as nx

from src.utils.paths import RESULT_DIR, PATHS_DIR
from src.utils.graph import calc_path_length, calc_path_similarity, load_network


def calc_avg_path_len_hops(
    graph: nx.Graph,
    paths: Dict[Tuple[int, int], List[Tuple[int]]]
    ) -> float:
    """
    パスの長さとホップ数の平均を計算する関数
    """
    length_sum = sum(
        sum([calc_path_length(graph, path) for path in paths[key]]) 
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
    paths: Dict[Tuple[int, int], List[Tuple[int]]]
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
    paths: Dict[Tuple[int, int], List[Tuple[int]]]
    ) -> float:
    """
    パスの数の平均を計算する関数
    """
    path_num = sum([len(paths[key]) for key in paths.keys()])
    path_num_ave = path_num / len(paths.keys())

    return path_num_ave


def calc_edge_usage_metrics(
    graph: nx.Graph, 
    paths: Dict[Tuple[int, int], List[Tuple[int]]]
    ) -> Dict[Tuple[int, int], int]:
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
    paths: Dict[Tuple[int, int], List[Tuple[int]]]
    ) -> Dict[str, Any]:
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
    network_names = ['NSF', 'EURO16', 'JPN12']
    algo_names = ['kSP-hop', 'kSP', 'kSPwLO', 'kDP']
    metrics = ['length_ave', 'hop_ave', 'similarity_ave']
    # initialize metrics_table
    for network_name in network_names:
        graph = load_network(network_name)
        metrics_table = pd.DataFrame()
        for k in range(2, 4):
            for algo_name in algo_names:
                if algo_name == 'kSPwLO':
                    for alpha in np.arange(0.1, 1.0, 0.1):
                        file_path = PATHS_DIR / network_name / algo_name / f'k={k}_alpha={round(alpha, 2)}.pickle'
                        with open(PATHS_DIR/ 'NSF' / file_path, 'rb') as f:
                            all_paths = pickle.load(f)
                        all_metrics = calc_all_metrics(graph, all_paths)
                        for metric in metrics:
                            metrics_table.loc[f"k={k}_{algo_name}_{round(alpha, 2)}", metric] = all_metrics[metric]
                else:
                    file_path = PATHS_DIR / network_name / algo_name / f'k={k}.pickle'
                    with open(PATHS_DIR / 'NSF' / file_path, 'rb') as f:
                        all_paths = pickle.load(f)
                    all_metrics = calc_all_metrics(graph, all_paths)
                    for metric in metrics:
                        metrics_table.loc[f"k={k}_{algo_name}", metric] = all_metrics[metric]

        full_table_path = RESULT_DIR / 'paths' / f'{network_name}_metrics_table.csv'
        metrics_table.to_csv(full_table_path)
