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
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_balanced_paths import KSPwithSimilarityConstraint
from src.paths.algorithms.hierarchical_clustering import HierarchicalClustering


def main():
    # dummy
    dummy = gp.Model('dummy')
    # parameters
    network_names = [
        # 'N6', 
        'N6S9', 
        # 'RING', 
        'NSF', 
        'EURO16', 
        # 'US24', 
        'JPN12', 
        # 'JPN25', 
        # 'GRID2x2', 
        # 'GRID2x3', 
        'GRID3x3', 
        'GRID3x4', 
        # 'GRID4x4', 
    ]
    n_paths_values      = [k for k in range(2, 4)]
    path_metric_list    = ['physical-length', 'hop', 'expected-used-slots']
    sim_metric_list     = ['physical-length', 'all-one']
    cls_distance_list   = ['single', 'average']
    alpha_values        = [round(i * 0.25, 2) for i in range(1, 4)]

    # k-shortest-paths
    print("k-Shortest Paths")
    for graph_name in tqdm.tqdm(network_names):
        for path_metric in tqdm.tqdm(path_metric_list, leave=False):
            for n_paths in tqdm.tqdm(n_paths_values, leave=False):
                k_shortest_paths = KShortestPaths(graph_name, n_path, {'path_weight': path_weight})
                k_shortest_paths.save_selected_paths_all_pairs()

    # # k-dissimilar-paths
    # print('k-dissimilar-paths')
    # for graph_name in tqdm.tqdm(network_names):
    #     for sim_weight in tqdm.tqdm(sim_weights, leave=False):
    #         for n_path in tqdm.tqdm(n_paths, leave=False):
    #             k_dissimilar_paths = KDissimilarPaths(graph_name, n_path, {'sim_weight': sim_weight})
    #             k_dissimilar_paths.save_selected_paths_all_pairs()

    # # k-balanced-paths
    # print('k-balanced-paths')
    # for graph_name in tqdm.tqdm(network_names, desc='graph'.ljust(15)):
    #     for n_path in tqdm.tqdm(n_paths, desc='n_paths'.ljust(15), leave=False):
    #         for path_weight in tqdm.tqdm(path_weights, desc='path_weight'.ljust(15), leave=False):
    #             for sim_weight in tqdm.tqdm(sim_weights, desc='sim_weight'.ljust(15), leave=False):
    #                 for alpha in tqdm.tqdm(alpha_values, desc='alpha'.ljust(15), leave=False):
    #                     k_balanced_paths = KSPwithSimilarityConstraint(
    #                         graph_name, n_path, {'path_weight': path_weight, 'sim_weight': sim_weight, 'alpha': alpha}
    #                         )
    #                     k_balanced_paths.save_selected_paths_all_pairs()
    
    # print('Hierarchical Clustering')
    # for graph_name in tqdm.tqdm(network_names, desc='graph'.ljust(15)):
    #     for n_path in tqdm.tqdm(n_paths, desc='n_paths'.ljust(15), leave=False):
    #         for path_weight in tqdm.tqdm(path_weights, desc='path_weight'.ljust(15), leave=False):
    #             for sim_weight in tqdm.tqdm(sim_weights, desc='sim_weight'.ljust(15), leave=False):
    #                 for cls_distance in tqdm.tqdm(cls_distances, desc='cls_distance'.ljust(15), leave=False):
    #                     hierarchical_clustering = HierarchicalClustering(
    #                         graph_name, n_path, {'path_weight': path_weight, 'sim_weight': sim_weight, 'cls_distance': cls_distance}
    #                         )
    #                     hierarchical_clustering.save_selected_paths_all_pairs()

if __name__ == '__main__':
    main()
