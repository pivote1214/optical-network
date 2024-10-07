import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import networkx as nx

from utils.namespaces import NETWORK_DIR


NETWORK_LIST = [
    'N6', 
    'N6S9', 
    'RING', 
    'NSF', 
    'EURO16', 
    'US24', 
    'JPN12', 
    'JPN25', 
    'GRID2x2', 
    'GRID2x3', 
    'GRID3x3', 
    'GRID3x4', 
    'GRID4x4', 
]

def load_network(network_name: str) -> nx.DiGraph:
    """
    csvファイルからトポロジーを読み込む

    Args:
        network (str): ネットワークの名前

    Returns:
        network (nx.DiGraph): ネットワークのグラフ
    """
    if network_name == 'test':
        network = difine_test_graph()
        return network
    elif network_name not in NETWORK_LIST:
        raise ValueError(f'network_name must be in {NETWORK_LIST}')
    # ファイルパスの設定
    full_path = os.path.join(NETWORK_DIR, f'{network_name}.csv')
    # csvファイルをdataframeとして読み込む
    network_df = pd.read_csv(full_path, dtype={'source': int, 'target': int, 'weight': float})
    # dataframeからグラフを作成
    network = nx.from_pandas_edgelist(network_df, edge_attr=['weight'])
    # 双方向グラフに変換
    network = nx.DiGraph(network)
    return network


def create_noisy_grid_graph(H: int, W: int, interval: int) -> nx.Graph:
    """
    グリッドグラフを作成する

    Args:
        H (int): 縦のノード数
        W (int): 横のノード数
        interval (int): ノード間の間隔

    Returns:
        grid (nx.DiGraph): グリッドグラフ
    """
    # グリッドのノードを配置
    np.random.seed(1214)
    nodes = []
    for i in range(W):
        for j in range(H):
            nodes.append((i * interval, j * interval))
    
    # 原点と最も遠い点を除いたノードにランダムノイズを追加
    noisy_nodes = []
    noise_std = interval * 0.05
    for x, y in nodes:
        if (x, y) != (0, 0) and (x, y) != ((W-1) * interval, (H-1) * interval):
            noisy_x = x + np.random.normal(0, noise_std)
            noisy_y = y + np.random.normal(0, noise_std)
            noisy_nodes.append((noisy_x, noisy_y))
        else:
            noisy_nodes.append((x, y))
    
    # グラフを作成
    grid = nx.grid_2d_graph(W, H, create_using=nx.Graph())
    
    # ノイズを加えたノードの座標をグラフのノードに設定
    pos = {i * H + j: noisy_nodes[i * H + j] for i in range(W) for j in range(H)}
    
    # ノード番号を振り直す
    mapping = {(i, j): i * H + j for i in range(W) for j in range(H)}
    grid = nx.relabel_nodes(grid, mapping)
    
    # エッジに重みを追加
    for (u, v) in grid.edges():
        u_pos = pos[u]
        v_pos = pos[v]
        weight = np.sqrt((u_pos[0] - v_pos[0]) ** 2 + (u_pos[1] - v_pos[1]) ** 2)
        grid[u][v]['weight'] = weight
    
    # 有向グラフに変換
    grid = nx.DiGraph(grid)
    
    return grid



def calc_path_weight(
    network_topology: nx.DiGraph, 
    path: list[int], 
    metrics: str = 'physical-length', 
    modulation_format: dict = {600: 4, 1200: 3, 3500: 2, 6300: 1}
    ) -> int:
    """
    パスの長さを計算する

    Args:
        network_topology (nx.DiGraph): ネットワークのグラフ
        path (list[int]): パス
        metrics (str): パスの重みを計算する際の指標
        modulation_format (dict): パスの長さと変調方式の対応表

    Returns:
        path_weight (int): パスの重み
    """
    physical_length = sum([network_topology[path[i]][path[i + 1]]['weight'] 
                           for i in range(len(path) - 1)])
    n_edges = len(path) - 1
    if metrics == 'physical-length':
        path_weight = sum([network_topology[path[i]][path[i + 1]]['weight']
                           for i in range(len(path) - 1)])
    elif metrics == 'hop':
        path_weight = len(path) - 1
    elif metrics == 'expected-used-slots':
        physical_length = sum([network_topology[path[i]][path[i + 1]]['weight'] 
                            for i in range(len(path) - 1)])
        n_edges = len(path) - 1
        modulation_level = select_modulation_format(physical_length, modulation_format)
        max_level = max(modulation_format.values())
        path_weight = n_edges * (max_level - modulation_level + 1)
    else:
        raise ValueError('metrics must be "physical-length", "hop" or "expected-used-slots"')
    return path_weight


def calc_path_similarity(
    graph: nx.DiGraph, 
    path1: list[int], 
    path2: list[int], 
    edge_weight: str = 'physical-length'
    ) -> float:
    """
    2つのパスの類似度を計算する
    """
    # パスをエッジの集合に変換
    edges_path1 = {(path1[i], path1[i + 1]) for i in range(len(path1) - 1)}
    edges_path2 = {(path2[i], path2[i + 1]) for i in range(len(path2) - 1)}

    # エッジの重みによって類似度を計算
    if edge_weight == 'physical-length':
        common_edges = sum([graph[a][b]['weight'] for a, b in edges_path1 & edges_path2])
        path1_edges = sum([graph[a][b]['weight'] for a, b in edges_path1])
        path2_edges = sum([graph[a][b]['weight'] for a, b in edges_path2])
    elif edge_weight == 'all-one':
        common_edges = len(edges_path1 & edges_path2)
        path1_edges = len(edges_path1)
        path2_edges = len(edges_path2)
    else:
        raise ValueError('edge_weight must be "physical-length" or "all-one"')

    # 類似度を計算
    similarity = common_edges / min(path1_edges, path2_edges)
    return similarity


def is_edge_in_path(path: list[int], edge: tuple[int, int]) -> bool:
    """judge whether edge is in path"""
    judge = False
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) == edge:
            judge = True

    return judge


def judge_common_edges(path1: list[int], path2: list[int]) -> bool:
    """judge whether two paths have common edges"""
    judge = False
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            if (path1[i], path1[i + 1]) == (path2[j], path2[j + 1]):
                judge = True

    return judge


def select_modulation_format(
    path_length: int, 
    modulation_format: dict = {600: 4, 1200: 3, 3500: 2, 6300: 1}
    ) -> int:
    """
    パスの長さから変調方式を選択

    Args:
        path_length (int): パスの長さ
        modulation_format (dict): パスの長さと変調方式の対応表

    Returns:
        modulation_level (int): 選択された変調方式のレベル
    """
    for length_limit, modulation_level in modulation_format.items():
        if path_length <= length_limit:
            return modulation_level
    raise ValueError("Path length is too long.")


# make graph pickle file
if __name__ == "__main__":
    # print('Enter network_topology name: ')
    # network_name = input()
    # network = load_network(network_name)
    H, W = 4, 4
    network = create_noisy_grid_graph(H, W, 50)
    print(f'Nodes: {list(network.nodes)}')
    print(f'Edges: {list(network.edges)}')
    print(f'Number of nodes: {network.number_of_nodes()}')
    print(f'Number of edges: {network.number_of_edges()}')
    # source,target,weight のcsvファイルを作成
    edge_list = []
    for (u, v) in network.edges():
        edge_list.append([u, v, network[u][v]['weight']])
    edge_df = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
    # weight列を小数点以下2桁で丸める
    edge_df['weight'] = edge_df['weight'].round(2)
    edge_df.to_csv(os.path.join(NETWORK_DIR, f'GRID{H}x{W}.csv'), index=False)


def difine_test_graph(draw: bool=False) -> nx.DiGraph:
    # ノードの座標の定義
    node_positions = {
        1: (0, 1),
        2: (1, 1),
        3: (2, 1),
        4: (0, 0),
        5: (1, 0)
    }

    # エッジの定義
    edges = [
        (1, 2), (1, 4), (1, 5),
        (2, 3), (2, 5), (3, 5),
        (4, 5)
    ]

    # グラフの構築
    graph = nx.Graph()
    graph.add_nodes_from(node_positions.keys())
    graph.add_edges_from(edges)

    # エッジの重み（ユークリッド距離）の計算
    for (u, v) in graph.edges():
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        distance = np.hypot(x2 - x1, y2 - y1)
        graph.edges[u, v]['weight'] = distance
    # グラフの描画
    if draw:
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        for key, value in edge_labels.items():
            edge_labels[key] = round(value, 2)
        nx.draw_networkx(graph, node_positions, with_labels=True, node_size=500)
        nx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels)
    # 双方向グラフ化
    graph = graph.to_directed()

    return graph
