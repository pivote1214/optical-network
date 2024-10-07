import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
    )

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster

from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from utils.network import calc_path_similarity, calc_path_weight


@dataclass
class HierarchicalClusteringParams:
    length_metric:  str
    sim_metric:     str
    linkage_method: str


class HierarchicalClustering(PathSelectionAlgorithm):
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: dict = {'path_weight': 'physical-length', 'sim_weight': 'physical-length', 'cls_distance': 'single'}, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, params, length_limit)

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ) -> list[tuple[int]]:
        # source から target までの全てのシンプルパスを取得
        all_simple_paths = self._calc_all_simple_paths(source, target)
        # 全パス間の距離行列を計算
        distance_matrix = self._calc_distance_matrix(all_simple_paths)
        # パスの重みを計算
        w_paths = np.array([calc_path_weight(self.graph, path, self.params['path_weight']) for path in all_simple_paths])
        # 階層型クラスタリングを実行
        Z = hierarchical_clustering(
            distance_matrix, self.params['cls_distance']
        )
        selected_paths_idxs = select_paths_idx(Z, w_paths, self.n_paths)
        if len(selected_paths_idxs) != self.n_paths:
            raise ValueError('Number of selected paths is not equal to n_paths')
        selected_paths = [all_simple_paths[idx] for idx in selected_paths_idxs]

        return selected_paths

    def _calc_distance_matrix(
        self, 
        all_simple_paths: list[list[int]]
        ) -> np.ndarray:
        # numpy配列の初期化
        N_PATH = len(all_simple_paths)
        distance_matrix = np.zeros((N_PATH, N_PATH))

        for path_idx in range(N_PATH):
            for path_jdx in range(path_idx + 1, N_PATH):
                distance_matrix[path_idx][path_jdx] = distance_matrix[path_jdx][path_idx] = \
                    1 - calc_path_similarity(self.graph, all_simple_paths[path_idx], 
                                             all_simple_paths[path_jdx], edge_weight=self.params['sim_weight'])

        return distance_matrix


def calc_cluster_distnce(
    cluster1: list[int], 
    cluster2: list[int], 
    distance_matrix: np.ndarray, 
    cls_distance: str='single'
    ) -> float:
    """
    Calculate distance between two clusters using distance matrix.
    """
    if cls_distance == 'single':
        return np.min(distance_matrix[cluster1][:, cluster2])
    elif cls_distance == 'complete':
        return np.max(distance_matrix[cluster1][:, cluster2])
    elif cls_distance == 'average':
        return np.mean(distance_matrix[cluster1][:, cluster2])
    else:
        raise ValueError(f"Invalid cls_distance: {cls_distance}")


def hierarchical_clustering(
    distance_matrix: np.ndarray, 
    cls_distance: str
    ) -> np.ndarray:
    """
    Perform hierarchical clustering.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix between samples.

    Returns
    -------
    Z : np.ndarray
        Linkage matrix. Each row represents a pair of clusters to be merged.
        The first two elements are the cls_indices of clusters to be merged,
        the third element is the distance between the clusters,
        and the fourth element is the number of samples in the new cluster.
    """
    N = new_cls_idx = len(distance_matrix)
    clusters = {idx: [idx] for idx in range(N)}
    Z = np.array([], dtype=float).reshape(0, 4)

    while len(clusters) > 1:
        # 最小距離を探す
        min_dist = float('inf')
        to_merge = None

        for cls1 in clusters:
            for cls2 in clusters:
                if cls1 != cls2:
                    dist = calc_cluster_distnce(
                        clusters[cls1], clusters[cls2], distance_matrix, 
                        cls_distance=cls_distance
                        )
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = cls1, cls2

        if not to_merge:
            break

        # クラスタを結合
        cls1, cls2 = to_merge
        new_cluster = clusters[cls1] + clusters[cls2]
        # 更新
        clusters[new_cls_idx] = new_cluster
        del clusters[cls1]
        del clusters[cls2]

        # リンク情報を保存
        Z = np.vstack([Z, [cls1, cls2, min_dist, len(new_cluster)]])
        new_cls_idx += 1

    return Z


def select_paths_idx(
    Z: np.ndarray, 
    w_paths: np.ndarray, 
    num_paths: int
    ) -> list[int]:
    """
    Select a path of clusters to form a specified number of clusters.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix.
    w_paths : np.ndarray
        List of path weights.
    num_paths : int
        Number of clusters to be selcted.

    Returns
    -------
    selected_paths : list[int]
        List of index of paths to be selected.
    """
    selected_paths = []
    # maxclustでクラスタリング
    maxclust = fcluster(Z, num_paths, criterion='maxclust')
    # maxclustで選択されたクラスタの数とnum_pathsを比較
    if len(set(maxclust)) == num_paths:
        # 各クラスタの重みの最も小さいものを選択
        for i in range(1, num_paths+1):
            # クラスタ番号がiのパスの中で重みが最小のものを選択
            path_idxs = np.where(maxclust == i)[0]
            selected_paths.append(path_idxs[np.argmin(w_paths[path_idxs])])
    else:
        # クラスタ数がnum_pathsより多くなるまでmaxclustを増やす
        maxclust_before = copy.deepcopy(maxclust)
        count = 1
        while len(set(maxclust)) <= num_paths:
            maxclust = fcluster(Z, num_paths+count, criterion='maxclust')
            count += 1
            if num_paths + count > Z.shape[0] + 1:
                maxclust = np.array(range(1, Z.shape[0] + 2), dtype=float)
                break
        n_before = len(set(maxclust_before))
        n_after = len(set(maxclust))
        # maxclust_beforeとmaxclustのクラスタが同じクラスタのパスを選ぶ
        for cls in range(1, n_before+1):
            # cls_idxのクラスタの中で重みが最小のものを選択
            path_idxs = np.where(maxclust_before == cls)[0]
            w_min_path = path_idxs[np.argmin(w_paths[path_idxs])]
            selected_paths.append(w_min_path)
            # 選ばれたパスのmaxclustにおけるクラスタ番号を-1にする
            cls_after = maxclust[w_min_path]
            maxclust[maxclust == cls_after] = -1

        # num_pathsの数までパスを選択
        for _ in range(num_paths - len(selected_paths)):
            w_min = float('inf')
            w_min_path = None
            selected_cls = None
            for cls in range(1, n_after+1):
                if cls not in maxclust:
                    continue
                path_idxs = np.where(maxclust == cls)[0]
                w = np.min(w_paths[path_idxs])
                if w < w_min:
                    w_min = w
                    w_min_path = path_idxs[np.argmin(w_paths[path_idxs])]
                    selected_cls = cls
            selected_paths.append(w_min_path)
            # 選ばれたクラスタ番号を-1にする
            maxclust[maxclust == selected_cls] = -1

    return selected_paths
                

# デンドログラムを描くための関数
def plot_dendrogram(Z: np.ndarray) -> None:
    dendrogram(Z)
    plt.xlabel('Path Index')
    plt.ylabel('Distance')
    plt.show()


if __name__ == '__main__':
    # 同じ距離を持つクラスタを含む距離行列 D の例
    D = np.array([
        [0, 6, 2, 4, 1, 7, 7, 2], 
        [6, 0, 8, 4, 7, 1, 1, 8], 
        [2, 8, 0, 4, 1, 9, 9, 0], 
        [2, 4, 4, 0, 3, 5, 5, 4], 
        [1, 7, 1, 3, 0, 8, 8, 1], 
        [7, 1, 9, 5, 8, 0, 0, 9], 
        [7, 1, 9, 5, 8, 0, 0, 9], 
        [2, 8, 0, 4, 1, 9, 9, 0]
    ], dtype=float)
    D /= 10
    D = D[[1, 5, 6, 3, 2, 7, 0, 4], :]
    D = D[:, [1, 5, 6, 3, 2, 7, 0, 4]]

    w_paths = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # 階層型クラスタリングを実行
    Z = hierarchical_clustering(D)
    print(Z)

    # パスを選択
    num_paths = 4
    selected_paths = select_paths_idx(Z, w_paths, num_paths)
    print(selected_paths)

    # デンドログラムを描画
    plot_dendrogram(Z)
