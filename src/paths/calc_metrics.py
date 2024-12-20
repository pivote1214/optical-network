import json
import os
import re
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.graph import calc_path_similarity, calc_path_weight
from src.namespaces import DATA_DIR

__all__ = [
    "collect_time_data",
    "calc_physical_length_all_paths",
    "calc_hop_all_paths",
    "calc_similarity_all_paths",
    "calc_edge_usage_all_paths",
    "calc_edge_usage_nodepair",
]


def collect_time_data(
    path_selector_name: str, network_name: str, n_paths: int
) -> NDArray[np.float64]:
    """
    パス選択アルゴリズムの計算時間を計算
    """
    name2cname = {
        "ksp": "kShortestPaths",
        "kspasc": "KSPAlphaSimilarityConstraint",
        "kdp": "kDissimilarPaths",
        "path_clst": "PathClustering",
        "nodepair_clst": "NodePairClustering",
    }
    cname = name2cname[path_selector_name]
    dir = os.path.join(DATA_DIR, "paths", cname, network_name)
    time_list = []
    # 正規表現パターンを作成
    pattern = re.compile(rf".*n-paths={n_paths}/time\.json$")
    # ディレクトリを再帰的に走査
    for root, _, files in os.walk(dir):
        for file in files:
            # ファイルのフルパスを作成
            full_path = os.path.join(root, file)
            # 正規表現でパスをチェック
            if pattern.match(full_path):
                with open(full_path, "r") as f:
                    time_list.append(json.load(f)["time"])
    return np.array(time_list)


def calc_physical_length_all_paths(
    graph: nx.DiGraph,
    all_paths: dict[tuple[Any, Any], list[tuple[Any]]],
) -> NDArray[np.float64]:
    """
    全てのパスの長さを計算
    """
    length_list = np.array(
        [
            calc_path_weight(graph, path, metrics="physical-length")
            for paths in all_paths.values()
            for path in paths
        ]
    )

    return length_list


def calc_hop_all_paths(
    graph: nx.DiGraph,
    all_paths: dict[tuple[Any, Any], list[tuple[Any]]],
) -> NDArray[np.float64]:
    """
    全てのパスのホップ数を計算
    """
    hop_list = np.array(
        [
            calc_path_weight(graph, path, metrics="hop")
            for paths in all_paths.values()
            for path in paths
        ]
    )
    return hop_list


def calc_similarity_all_paths(
    graph: nx.DiGraph,
    all_paths: dict[tuple[Any, Any], list[tuple[Any]]],
    metric: str,
) -> NDArray[np.float64]:
    """
    全てのパスペアの類似度を計算
    """
    similarity_list = np.array(
        [
            calc_path_similarity(graph, paths[pi], paths[pj], metric=metric)
            for paths in all_paths.values()
            for pi in range(len(paths))
            for pj in range(pi + 1, len(paths))
        ]
    )
    return similarity_list


def calc_edge_usage_all_paths(
    graph: nx.DiGraph,
    all_paths: dict[tuple[Any, Any], list[tuple[Any]]],
) -> dict[tuple[Any, Any], int]:
    """
    各辺の使用回数を計算
    """
    edge_usage = {(u, v): 0 for u, v in graph.edges()}
    for (u, v), paths in all_paths.items():
        if u >= v:
            continue
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                edge_usage[(u, v)] += 1
    return edge_usage


def calc_edge_usage_nodepair(
    graph: nx.DiGraph, 
    all_paths: dict[tuple[Any, Any], list[tuple[Any]]],
) -> dict[tuple[Any, Any], int]:
    """
    各辺が属するノードペアの数を計算
    """
    edge_usage = {(u, v): 0 for u, v in graph.edges()}
    for (u, v), paths in all_paths.items():
        if u >= v:
            continue
        is_count = set()
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                is_count.add((u, v))
        for edge in is_count:
            edge_usage[edge] += 1
    return edge_usage
