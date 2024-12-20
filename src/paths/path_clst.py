import copy
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from src.graph import (
    calc_all_simple_paths,
    calc_path_similarity,
    calc_path_weight,
)
from src.paths._registry import register_path_selector
from src.paths.base import BasePathSelectorSinglePair

__all__ = ["path_clst"]


class PathClustering(BasePathSelectorSinglePair):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int,
        length_metric: str = "hop",
        sim_metric: str = "physical-length",
        linkage_method: str = "single",
    ):
        super().__init__(
            graph,
            n_paths,
            max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            linkage_method=linkage_method,
        )
        self.length_metric = length_metric
        self.sim_metric = sim_metric
        self.linkage_method = linkage_method

    def select_paths_single_pair(self, source: Any, target: Any) -> list[tuple[Any]]:
        all_simple_paths = self.all_simple_paths[(source, target)]
        distance_vector = self._calc_distance_vector(all_simple_paths)
        # パスの重みを計算
        w_paths = np.array(
            [
                calc_path_weight(self.graph, path, self.length_metric)
                for path in all_simple_paths
            ]
        )
        # 階層型クラスタリングを実行
        Z = linkage(distance_vector, method=self.linkage_method)
        selected_paths_idxs = select_paths_idx(Z, w_paths, self.n_paths_per_pair[source, target])
        if len(selected_paths_idxs) != self.n_paths_per_pair[source, target]:
            raise ValueError("Number of selected paths is not equal to n_paths")
        selected_paths = [all_simple_paths[idx] for idx in selected_paths_idxs]

        return selected_paths

    def _calc_distance_vector(
        self, all_simple_paths: list[list[Any]]
    ) -> NDArray[np.float64]:
        """
        Calculate the condensed distance vector directly without constructing a full matrix.

        Parameters
        ----------
        all_simple_paths : list[list[Any]]
            List of simple paths to compute pairwise distances.

        Returns
        -------
        distance_vector : np.ndarray
            Condensed 1D distance vector suitable for scipy's linkage function.
        """
        # Calculate the pairwise distances directly in condensed form
        distance_vector = []

        # Use combinations to iterate over all unique pairs (i, j) with i < j
        for p_i in range(len(all_simple_paths)):
            for p_j in range(p_i + 1, len(all_simple_paths)):
                # Compute the similarity and convert it to distance
                similarity = calc_path_similarity(
                    self.graph,
                    all_simple_paths[p_i],
                    all_simple_paths[p_j],
                    metric=self.sim_metric,
                )
                # Append the distance (1 - similarity) to the vector
                distance_vector.append(1 - similarity)

        # Convert to numpy array
        return np.array(distance_vector)


def select_paths_idx(Z: np.ndarray, w_paths: np.ndarray, num_paths: int) -> list[int]:
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
    maxclust = fcluster(Z, num_paths, criterion="maxclust")
    # maxclustで選択されたクラスタの数とnum_pathsを比較
    if len(set(maxclust)) == num_paths:
        # 各クラスタの重みの最も小さいものを選択
        for i in range(1, num_paths + 1):
            # クラスタ番号がiのパスの中で重みが最小のものを選択
            path_idxs = np.where(maxclust == i)[0]
            selected_paths.append(path_idxs[np.argmin(w_paths[path_idxs])])
    else:
        # クラスタ数がnum_pathsより多くなるまでmaxclustを増やす
        maxclust_before = copy.deepcopy(maxclust)
        count = 1
        while len(set(maxclust)) <= num_paths:
            maxclust = fcluster(Z, num_paths + count, criterion="maxclust")
            count += 1
            if num_paths + count > Z.shape[0] + 1:
                maxclust = np.array(range(1, Z.shape[0] + 2), dtype=float)
                break
        n_before = len(set(maxclust_before))
        n_after = len(set(maxclust))
        # maxclust_beforeとmaxclustのクラスタが同じクラスタのパスを選ぶ
        for cls in range(1, n_before + 1):
            # cls_idxのクラスタの中で重みが最小のものを選択
            path_idxs = np.where(maxclust_before == cls)[0]
            w_min_path = path_idxs[np.argmin(w_paths[path_idxs])]
            selected_paths.append(w_min_path)
            # 選ばれたパスのmaxclustにおけるクラスタ番号を-1にする
            cls_after = maxclust[w_min_path]
            maxclust[maxclust == cls_after] = -1

        # num_pathsの数までパスを選択
        for _ in range(num_paths - len(selected_paths)):
            w_min = float("inf")
            w_min_path = None
            selected_cls = None
            for cls in range(1, n_after + 1):
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


@register_path_selector
def path_clst(
    graph: nx.DiGraph, n_paths: int, max_length: int, **kwargs: Any
) -> PathClustering:
    return PathClustering(graph, n_paths, max_length, **kwargs)
