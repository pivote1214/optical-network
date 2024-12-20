import numpy as np
import networkx as nx

__all__ = ["create_noisy_grid_graph"]


def create_noisy_grid_graph(height: int, width: int, interval: int) -> nx.Graph:
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
    for i in range(width):
        for j in range(height):
            nodes.append((i * interval, j * interval))
    
    # 原点と最も遠い点を除いたノードにランダムノイズを追加
    noisy_nodes = []
    noise_std = interval * 0.05
    for x, y in nodes:
        if (x, y) != (0, 0) and (x, y) != ((width-1) * interval, (height-1) * interval):
            noisy_x = x + np.random.normal(0, noise_std)
            noisy_y = y + np.random.normal(0, noise_std)
            noisy_nodes.append((noisy_x, noisy_y))
        else:
            noisy_nodes.append((x, y))
    
    # グラフを作成
    grid = nx.grid_2d_graph(width, height, create_using=nx.Graph())
    
    # ノイズを加えたノードの座標をグラフのノードに設定
    pos = {i * height + j: noisy_nodes[i * height + j] for i in range(width) for j in range(height)}
    
    # ノード番号を振り直す
    mapping = {(i, j): i * height + j for i in range(width) for j in range(height)}
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
