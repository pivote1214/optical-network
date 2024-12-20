import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.namespaces import NETWORK_DIR

__all__ = [
    "get_network_list",
    "load_network",
    "draw_network",
    "calc_all_simple_paths",
    "calc_path_weight",
    "calc_path_similarity",
    "calc_total_similarity",
    "is_edge_in_path",
    "judge_common_edges",
    "select_modulation_format",
]


def get_network_list() -> list[str]:
    """
    NETWORK_DIRに存在するcsvファイルのリストを取得する

    Returns
    -------
    network_list: list[str]
        定義済みのネットワークのリスト
    """
    network_list = []
    for file_name in os.listdir(NETWORK_DIR):
        if file_name.endswith(".csv"):
            network_list.append(file_name.replace(".csv", ""))
    return network_list


def load_network(network_name: str) -> nx.DiGraph:
    """
    csvファイルからグラフを作成

    Parameters
    ----------
    network_name: str
        ネットワーク名

    Returns
    -------
    network: nx.DiGraph
        ネットワークのグラフ
    """
    full_path = os.path.join(NETWORK_DIR, f"{network_name}.csv")
    network_df = pd.read_csv(
        full_path, dtype={"source": int, "target": int, "weight": float}
    )
    network = nx.from_pandas_edgelist(network_df, edge_attr=["weight"])
    # 双方向グラフに変換
    network = nx.DiGraph(network)
    network.name = network_name
    return network



def draw_network(
    graph: nx.DiGraph,
    is_edge_label: bool = False,
    is_edge_usage: bool = False, 
    title: str = None
    ) -> None:
    """
    グラフを描画

    Parameters
    ----------
    graph : nx.DiGraph
        グラフ
    is_edge_label : bool
        エッジの重みを表示するかどうか
    is_edge_usage : bool
        エッジの使用回数に基づいて色を付けるかどうか

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    # ノードの座標をcsvファイルから取得
    pos_file = os.path.join(NETWORK_DIR, "position", f"{graph.name}.csv")
    pos = pd.read_csv(pos_file, dtype={"node": int, "x": float, "y": float})
    pos = {node: (x, y) for node, x, y in pos.values}
    # ノードの描画
    nx.draw_networkx_nodes(
        graph, pos, node_color="lightblue", alpha=0.9, edgecolors="black", node_size=500
    )

    # エッジの描画と色付け
    if is_edge_usage:
        counts = [graph[u][v]["count"] for u, v in graph.edges()]
        max_count = max(counts)
        if max_count > 0:
            edge_colors = [plt.cm.Blues(count / max_count) for count in counts]
        else:
            edge_colors = "black"  # max_count が 0 の場合は全て黒にする
        nx.draw_networkx_edges(graph, pos, width=2.5, edge_color=edge_colors)
    else:
        nx.draw_networkx_edges(graph, pos, width=2.5, edge_color="black")

    # ノードラベルの描画
    nx.draw_networkx_labels(graph, pos, font_size=12)

    # エッジラベルの描画
    if is_edge_label:
        edge_labels = nx.get_edge_attributes(graph, "weight")
        edge_labels = {k: int(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=12, font_color="black", font_weight="bold"
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.show()


def calc_all_simple_paths(
    graph: nx.DiGraph,
    source: int,
    target: int,
    max_length: int,
    ) -> list[list[Any]]:
    all_simple_paths = list(nx.all_simple_paths(graph, source, target, max_length))
    # パスの重みが6300を超えるパスを削除
    all_simple_paths = [
        path for path in all_simple_paths if calc_path_weight(graph, path) <= 6300
        ]
    return all_simple_paths


def calc_path_weight(
    graph: nx.DiGraph,
    path: list[Any],
    metrics: str = "physical-length"
    ) -> int:
    """
    パスの重みを計算する

    Parameters
    ----------
    graph : nx.DiGraph
        ネットワークのグラフ
    path : list[int]
        パス
    metrics : str, optional
        パスの重みを計算する際の指標 ('physical-length' または 'hop')

    Returns
    -------
    path_weight : int
        パスの重み

    Raises
    ------
    ValueError
        metricsが'physical-length'または'hop'でない場合
    """
    if metrics == "physical-length":
        path_weight = sum(
            graph[path[i]][path[i + 1]]["weight"]
            for i in range(len(path) - 1)
        )
    elif metrics == "hop":
        path_weight = len(path) - 1
    else:
        raise ValueError('metrics must be "physical-length" or "hop"')
    return path_weight


def calc_path_similarity(
    graph: nx.DiGraph,
    path1: list[Any],
    path2: list[Any],
    metric: str = "physical-length",
    ) -> float:
    """
    2つのパスの類似度を計算する

    Parameters
    ----------
    graph : nx.DiGraph
        ネットワークのグラフ
    path1 : list[Any]
        比較する最初のパス
    path2 : list[Any]
        比較する2番目のパス
    metric : str, optional
        類似度計算に用いるエッジの重み

    Returns
    -------
    float
        2つのパスの類似度
    """
    # パスをエッジの集合に変換
    edges_path1 = {(path1[i], path1[i + 1]) for i in range(len(path1) - 1)}
    edges_path2 = {(path2[i], path2[i + 1]) for i in range(len(path2) - 1)}

    # エッジの重みによって類似度を計算
    if metric == "physical-length":
        common_edges = sum(
            [graph[a][b]["weight"] for a, b in edges_path1 & edges_path2]
        )
        path1_edges = sum([graph[a][b]["weight"] for a, b in edges_path1])
        path2_edges = sum([graph[a][b]["weight"] for a, b in edges_path2])
    elif metric == "all-one":
        common_edges = len(edges_path1 & edges_path2)
        path1_edges = len(edges_path1)
        path2_edges = len(edges_path2)
    else:
        raise ValueError('metric must be "physical-length" or "all-one"')

    # 類似度を計算
    similarity = common_edges / min(path1_edges, path2_edges)
    return similarity


def calc_total_similarity(
    graph: nx.DiGraph, 
    paths: list[list[Any]], 
    sim_metric: str
    ) -> float:
    """
    パスの集合の総類似度を計算する
    """
    total_similarity = sum(
        calc_path_similarity(graph, paths[i], paths[j], sim_metric) 
        for i in range(len(paths)) 
        for j in range(i + 1, len(paths))
    )
    return total_similarity

def is_edge_in_path(path: list[Any], edge: tuple[Any, Any]) -> bool:
    """
    パスにエッジが含まれているかを判定する

    Parameters
    ----------
    path : list[Any]
        パス
    edge : tuple[Any, Any]
        判定するエッジ

    Returns
    -------
    bool
        エッジがパスに含まれている場合はTrue、そうでない場合はFalse
    """
    judge = False
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) == edge:
            judge = True

    return judge


def judge_common_edges(path1: list[Any], path2: list[Any]) -> bool:
    """
    2つのパスに共通のエッジがあるかを判定する

    Parameters
    ----------
    path1 : list[Any]
        最初のパス
    path2 : list[Any]
        2番目のパス

    Returns
    -------
    bool
        共通のエッジがある場合はTrue、そうでない場合はFalse
    """
    judge = False
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            if (path1[i], path1[i + 1]) == (path2[j], path2[j + 1]):
                judge = True

    return judge


def select_modulation_format(
    path_length: int,
    modulation_format: list[tuple[int, int]] = [(600, 4), (1200, 3), (3500, 2), (6300, 1)],
    ) -> int:
    """
    パスの長さから変調方式を選択する

    Parameters
    ----------
    path_length : int
        パスの長さ
    modulation_format : dict, optional
        変調方式の最大伝送距離と変調レベル

    Returns
    -------
    int
        選択された変調方式のレベル

    Raises
    ------
    ValueError
        パスの長さが長すぎる場合
    """
    for length_limit, modulation_level in modulation_format:
        if path_length <= length_limit:
            return modulation_level
    raise ValueError("Path length is too long.")
