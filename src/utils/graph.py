from __future__ import annotations

from typing import List

import pickle
import networkx as nx

from src.utils.paths import GRAPH_DIR


# 各グラフの定義
KLI2018 = [(1, 2, 500), (1, 5, 500), (2, 3, 500), 
           (3, 4, 500), (3, 5, 809), (4, 5, 500)]

NSF = [(1, 2, 1100), (1, 3, 1600), (1, 8, 2800), (2, 3, 600), (2, 4, 1000), (3, 6, 2000), (4, 5, 600), 
       (4, 11, 2400), (5, 6, 1100), (5, 7, 800), (6, 10, 1200), (6, 13, 2000), (7, 8, 700), (8, 9, 700), 
       (9, 10, 900), (9, 12, 500), (9, 14, 500), (11, 12, 800), (11, 14, 800), (12, 13, 300), (13, 14, 300)]

N6 = [(1, 2, 1000), (1, 3, 1200), (2, 3, 600), (2, 4, 800), (2, 5, 1000), 
      (3, 5, 800), (4, 5, 600), (4, 6, 1000), (5, 6, 1200)]

N6S9 = [(0, 1, 1000), (0, 2, 1200), (1, 2, 600), 
        (1, 3, 800), (1, 4, 1000), (2, 4, 800),
        (3, 4, 600), (3, 5, 1000), (4, 5, 1200)]

RING = [(1, 2, 300), (2, 3, 230), (3, 4, 421), (4, 5, 323), 
        (5, 6, 432), (6, 7, 272), (7, 8, 297), (8, 1, 388)]

cr_table_network = {
                    'KLI2018': KLI2018,
                    'NSF': NSF, 
                    'N6': N6, 
                    'N6S9': N6S9, 
                    'RING': RING
                    }

def load_network(network_name: str) -> nx.Graph:
    """load graph from pickle file"""
    full_path = GRAPH_DIR / f"{network_name}.pickle"
    with open(full_path, 'rb') as f:
        graph = pickle.load(f)
        
    return graph

def create_network(network_name: str) -> nx.Graph:
    """
    グラフを作成する関数
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(cr_table_network[network_name])

    return graph

def calc_path_length(
    graph: nx.Graph, 
    path: List[int]
    ) -> int:
    """
    与えられたパスの長さを計算する関数
    """
    return sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))


def path_similarity(
    graph: nx.Graph, 
    path1: List[int], 
    path2: List[int]
    ) -> float:
    """
    2つのパスの類似度を計算する関数
    """
    edges_path1 = {(path1[i], path1[i + 1]) for i in range(len(path1) - 1)}
    edges_path2 = {(path2[i], path2[i + 1]) for i in range(len(path2) - 1)}

    edges_path1.update({(a, b) for b, a in edges_path1})
    edges_path2.update({(a, b) for b, a in edges_path2})

    common_edges = sum([graph[a][b]['weight'] for a, b in edges_path1 & edges_path2])
    path1_edges = sum([graph[a][b]['weight'] for a, b in edges_path1])
    path2_edges = sum([graph[a][b]['weight'] for a, b in edges_path2])

    if min(path1_edges, path2_edges) == 0:
        similarity = 0
    else:
        similarity = common_edges / min(path1_edges, path2_edges)

    return similarity


def is_edge_in_path(path: List[int], edge: tuple[int, int]) -> bool:
    """
    エッジがパスに含まれるかどうかを判定する関数
    """
    judge = False
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) == edge or (path[i + 1], path[i]) == edge:
            judge = True

    return judge


# graphのpickleファイルの作成
if __name__ == "__main__":
    for network_name in cr_table_network.keys():
        graph = create_network(network_name)
        full_path = GRAPH_DIR / f"{network_name}.pickle"
        with open(full_path, 'wb') as f:
            pickle.dump(graph, f)
