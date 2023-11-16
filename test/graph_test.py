import sys
sys.path.append('..')

import time
import tqdm
from src.paths.algorithms.k_dissimilar_paths import KDissimilarPaths
from src.paths.algorithms.k_shortest_paths import KShortestPaths
from src.paths.algorithms.k_balanced_paths import KBalancedPath

graph_name = 'NSF'
path_nums_list = [i for i in range(1, 11)]
alpha_list = [i * 0.01 for i in range(1, 10)]

for path_nums in tqdm.tqdm(path_nums_list):
    for alpha in tqdm.tqdm(alpha_list, leave=False):
        start = time.time()
        print("--------------------")
        print('path_nums: {}, alpha: {}'.format(path_nums, alpha))
        print('k_dissimilar_paths')
        k_dissimilar_paths = KDissimilarPaths(graph_name, path_nums, alpha)
