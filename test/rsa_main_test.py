import os
import sys

sys.path.append(os.path.abspath('..'))

import datetime

from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer
from src.optimize.models.RSA_PATH_CHANNEL.params import Parameter, TimeLimit, Width
from src.paths.algorithms.k_shortest_paths import KShortestPathsParams
from utils.network import load_network
from utils.files import set_paths_file_path, set_result_dir


path_params = KShortestPathsParams(length_metric='hop')
paths_file_path = set_paths_file_path('KShortestPaths', 'JPN12', path_params, n_paths=2)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
experiment_name = f'{timestamp}_rsa_main_test'
result_dir = set_result_dir(experiment_name, 'RSA_PATH_CHANNEL', 'JPN12', path_params, n_paths=2)
graph = load_network('JPN12')
timelimit = TimeLimit(lower=30.0, upper=150.0, main=720.0)
width = Width(OC=37.5, GB=6.25, FS=12.5)

test_data = [59, 76, 78, 81, 77]

os.makedirs(result_dir)

result = []
for seed in range(2, 11, 2):
    optimizer_params = Parameter(
        network_name='JPN12', 
        graph=load_network('JPN12'), 
        num_slots=85, 
        num_demands=100, 
        demands_population=[50, 100, 150, 200], 
        demands_seed=seed, 
        paths_dir=paths_file_path, 
        result_dir=result_dir, 
        bound_algo='without', 
        timelimit=timelimit, 
        width=width
    )
    optimizer = PathChannelOptimizer(optimizer_params)
    main_output, _, _ = optimizer.run()
    result.append(int(main_output.used_slots))
    print(int(main_output.used_slots))

if result == test_data:
    print('Test passed!')
else:
    print('Test failed!')
