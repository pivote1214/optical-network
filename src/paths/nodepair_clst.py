import os
from itertools import combinations, islice
from typing import Any

import networkx as nx
from gurobipy import GRB, Model, Var, quicksum
from scipy.cluster.hierarchy import fcluster, linkage

from src.graph import (
    calc_all_simple_paths,
    calc_path_similarity,
    calc_path_weight,
)
from src.paths._registry import register_path_selector
from src.paths.base import BasePathSelector

__all__ = ["nodepair_clst"]


class NodePairClustering(BasePathSelector):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int,
        length_metric: str,
        sim_metric: str,
        n_ref_paths: int,
        linkage_method: str,
        threshold: float,
        beta: float,
    ):
        super().__init__(
            graph,
            n_paths,
            max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            n_ref_paths=n_ref_paths,
            linkage_method=linkage_method,
            threshold=threshold,
            beta=beta,
        )
        self.length_metric = length_metric
        self.sim_metric = sim_metric
        self.n_ref_paths = n_ref_paths
        self.linkage_method = linkage_method
        self.threshold = threshold
        self.beta = beta

    def _select_all_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        clusters = self._node_pair_clustering()
        selected_paths = self._select_path_set_from_clusters(clusters)
        return selected_paths

    def _node_pair_clustering(self) -> dict[int, list[tuple[Any]]]:
        """ノードペア間のパスを基にクラスタリング"""
        # 全ノードペアを生成
        node_pairs = list(combinations(self.graph.nodes, 2))
        # 各ノードペアに対してクラスタリングの参考とするパスを取得
        node_pair_paths = {}
        for u, v in node_pairs:
            ref_paths = list(
                islice(
                    nx.shortest_simple_paths(self.graph, source=u, target=v),
                    self.n_ref_paths,
                )
            )
            node_pair_paths[(u, v)] = ref_paths
        # ノードペアの数を取得
        n_node_pairs = len(node_pairs)
        # 距離行列
        distance_vector = []
        for i in range(n_node_pairs):
            for j in range(i + 1, n_node_pairs):
                paths_i = node_pair_paths[node_pairs[i]]
                paths_j = node_pair_paths[node_pairs[j]]
                similarity = self._calc_node_pair_similarity(
                    paths_i, paths_j, edge_weight=self.sim_metric
                )
                distance = 1 - similarity  # 距離に変換
                distance_vector.append(distance)

        # 階層型クラスタリングを実行
        linked = linkage(distance_vector, method=self.linkage_method)
        # デンドログラムを参考にクラスタリング
        cluster_assignments = fcluster(linked, t=self.threshold, criterion="distance")
        # クラスタリング結果を辞書にまとめる
        clusters = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            pair = node_pairs[idx]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pair)

        return clusters

    def _select_path_set_from_clusters(
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
        pair2all_simple_paths = {}
        path_idx = 0
        path2idx = {}
        path2length = {}
        for u, v in node_pairs:
            # 選択の母集団となるパス集合の生成
            all_simple_paths = self.all_simple_paths[(u, v)]
            pair2all_simple_paths[(u, v)] = all_simple_paths
            # パスにインデックスを付与
            for path in all_simple_paths:
                path2idx[tuple(path)] = path_idx
                path_idx += 1
                path2length[tuple(path)] = calc_path_weight(
                    self.graph, path, metrics=self.length_metric
                )

        # ILP定式化
        model = Model("nodepair_clustering")
        # log to console: off, log file: on
        model.setParam("LogToConsole", 0)
        model.setParam(GRB.Param.Threads, 6)
        model.setParam("LogFile", os.path.join(self.paths_dir, "ilp.log"))
        # set time limit
        model.setParam(GRB.Param.TimeLimit, 600.0)
        # 変数の定義
        # x_{uvp} の定義
        x_uvp = {}
        for u, v in node_pairs:
            for path in pair2all_simple_paths[(u, v)]:
                path_tuple = tuple(path)
                var_name = f"x_{u}_{v}_{path2idx[path_tuple]}"
                x_uvp[(u, v, path_tuple)] = model.addVar(
                    vtype=GRB.BINARY, name=var_name
                )
        # z_{ce} の定義
        z_ce = {}
        for c in clusters:
            for e in self.graph.edges():
                var_name = f"z_{c}_{e[0]}_{e[1]}"
                z_ce[(c, e)] = model.addVar(vtype=GRB.BINARY, name=var_name)
        model.update()

        # 目的関数の定義
        model.setObjective(
            self.beta * quicksum(
                path2length[p_tuple] * x_uvp[(u, v, p_tuple)]
                for u, v in node_pairs
                for p_tuple in [tuple(p) for p in pair2all_simple_paths[(u, v)]]
                )
            + (1 - self.beta) * quicksum(
                z_ce[(c, e)] for c in clusters for e in self.graph.edges()
                ),
            GRB.MINIMIZE,
        )

        # 制約条件 1：それぞれの頂点対でちょうどk本のパスを選択
        for u, v in node_pairs:
            model.addConstr(
                quicksum(
                    x_uvp[(u, v, tuple(p))]
                    for p in pair2all_simple_paths[(u, v)]
                )
                == self.n_paths_per_pair[(u, v)],
                name=f"constraint_k_{u}_{v}",
            )

        # 制約条件 2：エッジ使用とクラスタ内リンク共有の関係
        for u, v in node_pairs:
            c = pair2cluster[(u, v)]
            for p in pair2all_simple_paths[(u, v)]:
                p_edges = [
                    (p[i], p[i + 1])
                    if (p[i], p[i + 1]) in self.graph.edges()
                    else (p[i + 1], p[i])
                    for i in range(len(p) - 1)
                ]
                for e in p_edges:
                    model.addConstr(
                        x_uvp[(u, v, tuple(p))] <= z_ce[(c, e)],
                        name=f"edge_usage_{u}_{v}_{path2idx[tuple(p)]}_{e}",
                    )

        # 制約条件 3: 各リンクは必ず使用される
        for e in self.graph.edges():
            model.addConstr(
                quicksum(z_ce[(c, e)] for c in clusters) >= 1,
                name=f"edge_usage_{e}",
            )

        # 求解
        model.optimize()

        # 選択されたパスを格納
        selected_paths = {}
        for u, v in node_pairs:
            selected_paths[(u, v)] = []
            selected_paths[(v, u)] = []
            for path in pair2all_simple_paths[(u, v)]:
                path_tuple = tuple(path)
                var: Var = x_uvp[(u, v, path_tuple)]
                if var.X > 0.5:
                    selected_paths[(u, v)].append(path)
                    selected_paths[(v, u)].append(list(reversed(path)))

        return selected_paths

    def _calc_node_pair_similarity(
        self, paths1: list[list[int]], paths2: list[list[int]], edge_weight: str
    ) -> float:
        """ノードペアの類似度を計算"""
        # グラフの作成
        bipartite_graph = nx.Graph()
        # 各パスの組み合わせに対して類似度を計算し、エッジとして追加
        for i, path1 in enumerate(paths1):
            for j, path2 in enumerate(paths2):
                similarity = calc_path_similarity(self.graph, path1, path2, edge_weight)
                # ノードを paths1 と paths2 で区別するために (i, j) 形式で管理
                bipartite_graph.add_edge(f"path1_{i}", f"path2_{j}", weight=similarity)

        # 最大重みマッチングを取得
        matching = nx.max_weight_matching(bipartite_graph, maxcardinality=True)
        # マッチングの合計重みを計算
        total_similarity = sum(bipartite_graph[u][v]["weight"] for u, v in matching)
        # 類似度の正規化
        normalized_similarity = total_similarity / min(len(paths1), len(paths2))

        return normalized_similarity


@register_path_selector
def nodepair_clst(
    graph: nx.DiGraph, n_paths: int, max_length: int, **kwargs
) -> NodePairClustering:
    return NodePairClustering(graph, n_paths, max_length, **kwargs)
