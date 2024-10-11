import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
    )

import tqdm

import gurobipy as gp

from src.paths.algorithms.k_shortest_paths import KShortestPaths, KShortestPathsParams
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths, KDissimilarPathsParams
from src.paths.algorithms.k_balanced_paths import KSPwithSimilarityConstraint, KSPwithSimilarityConstraintParams
from src.paths.algorithms.hierarchical_clustering import HierarchicalClustering, HierarchicalClusteringParams
from src.paths.algorithms.overall_optimization import NodePairClustering, NodePairClusteringParams


def main():
    # dummy
    dummy = gp.Model('dummy')
    # parameters
    graph_name_list = [
        # 'N6', 
        # 'N6S9', 
        # 'RING', 
        'NSF', 
        'JPN12', 
        'GRID3x3', 
        'GRID3x4', 
        # 'EURO16', 
        # 'US24', 
        # 'JPN25', 
        # 'GRID2x2', 
        # 'GRID2x3', 
        # 'GRID4x4', 
    ]
    n_paths_values      = [k for k in range(2, 4)]
    length_metric_list  = [
        # 'physical-length', 
        'hop', 
        # 'expected-used-slots'
        ]
    sim_metric_list     = [
        'physical-length', 
        'all-one'
        ]
    alpha_values        = [round(i * 0.25, 2) for i in range(1, 4)]
    linkage_method_list = ['single', 'average']
    n_ref_paths         = 1
    cutoff              = None
    citerion            = 'distance'
    threshold_values    = [round(i * 0.25, 2) for i in range(3, 4)]
    w_obj_values        = [round(i * 0.5, 1) for i in range(3)]

    # k-Shortest Paths
    print("k-Shortest Paths")
    for graph_name in tqdm.tqdm(graph_name_list):
        for length_metric in tqdm.tqdm(length_metric_list, leave=False):
            for n_paths in tqdm.tqdm(n_paths_values, leave=False):
                params = KShortestPathsParams(length_metric=length_metric)
                path_generator = KShortestPaths(graph_name, n_paths, params)
                path_generator.save_selected_paths()

    # # k-Dissimilar Paths
    # print('k-Dissimilar Paths')
    # for graph_name in tqdm.tqdm(graph_name_list):
    #     for sim_metric in tqdm.tqdm(sim_metric_list, leave=False):
    #         for n_paths in tqdm.tqdm(n_paths_values, leave=False):
    #             params = KDissimilarPathsParams(sim_metric=sim_metric)
    #             path_generator = KDissimilarPaths(graph_name, n_paths, params)
    #             path_generator.save_selected_paths()

    # k-Shortest Paths with Similarity Constraint
    print('k-Shortest Paths with Similarity Constraint')
    for graph_name in tqdm.tqdm(graph_name_list):
        for length_metric in tqdm.tqdm(length_metric_list, leave=False):
            for sim_metric in tqdm.tqdm(sim_metric_list, leave=False):
                for alpha in tqdm.tqdm(alpha_values, leave=False):
                    for n_paths in tqdm.tqdm(n_paths_values, leave=False):
                        params = KSPwithSimilarityConstraintParams(
                            length_metric=length_metric, 
                            sim_metric=sim_metric, 
                            alpha=alpha
                            )
                        path_generator = KSPwithSimilarityConstraint(graph_name, n_paths, params)
                        path_generator.save_selected_paths()

    # Hierarchical Clustering
    print('Hierarchical Clustering')
    for graph_name in tqdm.tqdm(graph_name_list, desc='graph'.ljust(15)):
        for length_metric in tqdm.tqdm(length_metric_list, desc='length_metric'.ljust(15), leave=False):
            for sim_metric in tqdm.tqdm(sim_metric_list, desc='sim_metric'.ljust(15), leave=False):
                for linkage_method in tqdm.tqdm(linkage_method_list, desc='linkage_method'.ljust(15), leave=False):
                    for n_paths in tqdm.tqdm(n_paths_values, desc='n_paths'.ljust(15), leave=False):
                        params = HierarchicalClusteringParams(
                            length_metric=length_metric, 
                            sim_metric=sim_metric, 
                            linkage_method=linkage_method
                            )
                        path_generator = HierarchicalClustering(graph_name, n_paths, params)
                        path_generator.save_selected_paths()

    # Node Pair Clustering
    print('Node Pair Clustering')
    for graph_name in tqdm.tqdm(graph_name_list, desc='graph'.ljust(15)):
        for length_metric in tqdm.tqdm(length_metric_list, desc='length_metric'.ljust(15), leave=False):
            for sim_metric in tqdm.tqdm(sim_metric_list, desc='sim_metric'.ljust(15), leave=False):
                for linkage_method in tqdm.tqdm(linkage_method_list, desc='linkage_method'.ljust(15), leave=False):
                    for threshold in tqdm.tqdm(threshold_values, desc='threshold'.ljust(15), leave=False):
                        for w_obj in tqdm.tqdm(w_obj_values, desc='w_obj'.ljust(15), leave=False):
                            for n_paths in tqdm.tqdm(n_paths_values, desc='n_paths'.ljust(15), leave=False):
                                params = NodePairClusteringParams(
                                    length_metric=length_metric, 
                                    sim_metric=sim_metric, 
                                    n_ref_paths=n_ref_paths, 
                                    cutoff=cutoff, 
                                    linkage_method=linkage_method, 
                                    criterion=citerion, 
                                    threshold=threshold, 
                                    w_obj=w_obj, 
                                    timelimit=600.0
                                    )
                                path_generator = NodePairClustering(graph_name, n_paths, params)
                                path_generator.save_selected_paths()


if __name__ == '__main__':
    main()
