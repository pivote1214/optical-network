import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
    )

from dataclasses import dataclass
from itertools import combinations, islice

import networkx as nx
import numpy as np
from gurobipy import GRB, Model, Var, quicksum
from scipy.cluster.hierarchy import fcluster, linkage

from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from utils.network import calc_path_similarity, calc_path_weight, load_network


@dataclass
class NodePairClusteringParams:
    length_metric:  str
    sim_metric:     str
    n_ref_paths:    int
    cutoff:         int
    linkage_method: str
    criterion:      str
    threshold:      float
    w_obj:          float


class NodePairClustering(PathSelectionAlgorithm):
    def __init__(
        self,
        graph_name: str, 
        n_paths: int, 
        params: NodePairClusteringParams, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, params, length_limit)
        self.params = params

    def select_k_paths_all_pairs(self) -> dict[tuple[int, int], list[tuple[int]]]:
        clusters = self.node_pair_clustering()
        selected_paths = self.select_path_set_from_clusters(clusters)
        return selected_paths

    def node_pair_clustering(self) -> dict[int, list[tuple[int]]]:
        """ノードペア間のパスを基にクラスタリング"""
        # 全ノードペアを生成
        node_pairs = list(combinations(self.graph.nodes, 2))
        # 各ノードペアに対してクラスタリングの参考とするパスを取得
        node_pair_paths = {}
        for u, v in node_pairs:
            ref_paths = list(islice(nx.shortest_simple_paths(self.graph, source=u, target=v), self.params.n_ref_paths))
            node_pair_paths[(u, v)] = ref_paths
        # ノードペアの数を取得
        n_node_pairs = len(node_pairs)
        # 距離行列
        distance_vector = []
        for i in range(n_node_pairs):
            for j in range(i + 1, n_node_pairs):
                paths_i = node_pair_paths[node_pairs[i]]
                paths_j = node_pair_paths[node_pairs[j]]
                similarity = self._calc_node_pair_similarity(paths_i, paths_j, edge_weight=self.params.sim_metric)
                distance = 1 - similarity  # 距離に変換
                distance_vector.append(distance)

        # 階層型クラスタリングを実行
        linked = linkage(distance_vector, method=self.params.linkage_method)
        # デンドログラムを参考にクラスタリング
        cluster_assignments = fcluster(linked, t=self.params.threshold, criterion=self.params.criterion)
        # クラスタリング結果を辞書にまとめる
        clusters = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            pair = node_pairs[idx]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pair)

        return clusters

    def select_path_set_from_clusters(
        self, 
        clusters: dict[int, list[tuple[int]]]
        ) -> dict[tuple[int, int], list[tuple[int]]]:
        """クラスタリング結果をもとにパスを選択"""
        # 定数の定義
        pair2cluster = {}
        for c_idx in clusters:
            for pair in clusters[c_idx]:
                pair2cluster[pair] = c_idx
        # 全ての頂点対のリスト
        node_pairs = list(combinations(self.graph.nodes(), 2))
        # 各頂点対に対して最短パスと候補パスを生成
        pair2shortest_path = {}
        pair2mother_set = {}
        path_idx = 0
        path2idx = {}
        for u, v in node_pairs:
            # 最短パスの計算
            shortest_path = nx.dijkstra_path(self.graph, u, v, weight='weight')
            pair2shortest_path[(u, v)] = shortest_path
            # 選択の母集団となるパス集合の生成
            mother_set = nx.all_simple_paths(self.graph, u, v, cutoff=self.params.cutoff)
            mother_set = [path for path in mother_set if path != shortest_path] # 最短パスを除く
            pair2mother_set[(u, v)] = mother_set
            # パスにインデックスを付与
            for path in mother_set:
                path2idx[tuple(path)] = path_idx
                path_idx += 1
        # パスの長さの比率 ratio_uvp の計算
        ratio_uvp = {}
        for u, v in node_pairs:
            shortest_length = calc_path_weight(self.graph, pair2shortest_path[(u, v)], metrics=self.params.length_metric)
            for path in pair2mother_set[(u, v)]:
                path_length = calc_path_weight(self.graph, path, metrics=self.params.length_metric)
                ratio_uvp[(u, v, tuple(path))] = path_length / shortest_length

        # ILP定式化
        model = Model("node-pair-clustering-method")
        # 変数の定義
        # x_{uvp} の定義
        x_uvp = {}
        for u, v in node_pairs:
            for path in pair2mother_set[(u, v)]:
                path_tuple = tuple(path)
                var_name = f"x_{u}_{v}_{path2idx[path_tuple]}"
                x_uvp[(u, v, path_tuple)] = model.addVar(vtype=GRB.BINARY, name=var_name)
        # z_{ce} の定義
        z_ce = {}
        for c in clusters:
            for e in self.graph.edges():
                var_name = f"z_{c}_{e[0]}_{e[1]}"
                z_ce[(c, e)] = model.addVar(vtype=GRB.BINARY, name=var_name)
        model.update()
        
        # 目的関数の定義
        model.setObjective(
            self.params.w_obj * quicksum(
                ratio_uvp[(u, v, p_tuple)] * x_uvp[(u, v, p_tuple)] 
                for u, v in node_pairs 
                for p_tuple in [tuple(p) for p in pair2mother_set[(u, v)]]
                ) +
            (1 - self.params.w_obj) * quicksum(
                z_ce[(c, e)] 
                for c in clusters 
                for e in self.graph.edges()
                ),
            GRB.MINIMIZE
        )
        
        # 制約条件 1：全ての頂点対で (k - 1) 本のパスを選択
        model.addConstr(
            quicksum(
                x_uvp[(u, v, tuple(p))] 
                for u, v in node_pairs
                for p in pair2mother_set[(u, v)]
                ) == (self.n_paths - 1) * len(node_pairs),  
            name="constraint_k"
        )
        
        # 制約条件 2：エッジ使用とクラスタ内リンク共有の関係
        for u, v in node_pairs:
            c = pair2cluster[(u, v)]
            for p in pair2mother_set[(u, v)]:
                p_edges = [(p[i], p[i+1]) if (p[i], p[i+1]) in self.graph.edges() else (p[i+1], p[i]) for i in range(len(p)-1)]
                for e in p_edges:
                    model.addConstr(
                        x_uvp[(u, v, tuple(p))] <= z_ce[(c, e)],
                        name=f"edge_usage_{u}_{v}_{path2idx[tuple(p)]}_{e}"
                    )
        
        # 求解
        model.optimize()
        
        # 選択されたパスを格納
        selected_paths = {}
        for u, v in node_pairs:
            selected_paths[(u, v)] = []
            selected_paths[(v, u)] = []
            for path in pair2mother_set[(u, v)]:
                path_tuple = tuple(path)
                var: Var = x_uvp[(u, v, path_tuple)]
                if var.X > 0.5:
                    selected_paths[(u, v)].append(path)
                    selected_paths[(v, u)].append(list(reversed(path)))
            # 最短パスを追加
            selected_paths[(u, v)].insert(0, pair2shortest_path[(u, v)])
            selected_paths[(v, u)].insert(0, list(reversed(pair2shortest_path[(u, v)])))

        return selected_paths

    def _calc_node_pair_similarity(
        self, 
        paths1: list[list[int]], 
        paths2: list[list[int]], 
        edge_weight: str
        ) -> float:
        """ノードペア間のパスの類似度を計算 (0から1)"""
        # パスの類似度の平均
        similarity = np.mean([calc_path_similarity(self.graph, path1, path2, edge_weight) 
                              for path1 in paths1 for path2 in paths2])
        return similarity
