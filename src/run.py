import os
import sys
sys.path.append(os.path.abspath('..'))

import time
import copy

import tqdm
import pandas as pd
import gurobipy as gp

from utils.network import load_network
from utils.namespaces import OUT_DIR
from optimize.models.RSA_PATH_CHANNEL.params import Parameter, TimeLimit, Width
from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer
from utils.files import set_paths_file_path, set_result_dir


if __name__ == "__main__":
    # dummy
    dummy = gp.Model('dummy')
    # experiment number
    experiment_name = "test"
    # set parameters (RSA-parameter)
    model_name              = 'RSA_PATH_CHANNEL'
    num_slots               = 320
    num_demands             = 5
    demands_population      = [50, 100, 150, 200]
    demands_seeds_values    = [seed * 2 for seed in range(1, 11)]
    # set parameters (Path-parameter)
    network_names  = ['JPN12', 'NSF', 'EURO16']
    path_algo_list          = ['k-shortest-paths', 
                               'k-dissimilar-paths', 
                               'k-shortest-paths-with-similarity-constraint', 
                               'hierarchical-clustering']
    path_weight_list        = ['hop', 'expected-used-slots']
    sim_weight_list         = ['physical-length', 'all-one']
    cls_distance_list       = ['single', 'average']
    alpha_list              = [round(i * 0.25, 2) for i in range(1, 4)]
    n_paths_list            = [2, 3]
    bound_algo              = "hybrid"
    timelimit               = TimeLimit(lower=30.0, upper=150.0, main=720.0)
    width                   = Width(OC=37.5, GB=6.25, FS=12.5)
    TRAFFIC_BPSK            = 50

    # make directory
    EX_DIR = os.path.join(OUT_DIR, experiment_name)
    os.makedirs(EX_DIR, exist_ok=True)

    # write global config
    CONFIG_FILE_NAME = os.path.join(EX_DIR, 'global_config.txt')
    with open(CONFIG_FILE_NAME, 'w') as f:
        f.write('global config\n')
        f.write(f'num_slots:            {num_slots}\n')
        f.write(f'num_demands:          {num_demands}\n')
        f.write(f'demands_population:   {demands_population}\n')
        f.write(f'bound_algo:           {bound_algo}\n')

    # run
    for network_name in tqdm.tqdm(network_names, desc='network'.ljust(20)):
        graph = load_network(network_name)
        for path_algo in tqdm.tqdm(path_algo_list, desc='path_algo'.ljust(20), leave=False):
            for n_paths in tqdm.tqdm(n_paths_list, desc='n_paths'.ljust(20), leave=False):
                for cls_distance in tqdm.tqdm(cls_distance_list, desc='cls_distance'.ljust(20), leave=False):
                    for alpha in tqdm.tqdm(alpha_list, desc='alpha'.ljust(20), leave=False):
                        for sim_weight in tqdm.tqdm(sim_weight_list, desc='sim_weight'.ljust(20), leave=False):
                            for path_weight in tqdm.tqdm(path_weight_list, desc='path_weight'.ljust(20), leave=False):
                                # set file path
                                params = {
                                    'path_weight': path_weight, 
                                    'sim_weight': sim_weight, 
                                    'alpha': alpha, 
                                    'cls_distance': cls_distance
                                    }
                                PATHS_FILE = set_paths_file_path(
                                    algorithm=path_algo, 
                                    network_name=network_name, 
                                    params=params, 
                                    n_paths=n_paths
                                    )
                                RESULT_DIR = set_result_dir(
                                    experiment_name=experiment_name, 
                                    algorithm=path_algo, 
                                    network_name=network_name, 
                                    params=params, 
                                    n_paths=n_paths
                                    )
                                # RESULT_FILE が存在するか判定
                                RESULT_FILE = os.path.join(RESULT_DIR, 'summary.csv')
                                if os.path.exists(RESULT_FILE):
                                    continue
                                else:
                                    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
                                # make table
                                index = copy.deepcopy(demands_seeds_values)
                                columns = [
                                    'used_slots',  'Gap(main)',  'time(main)', 
                                    'lower_bound', 'Gap(lower)', 'time(lower)', 
                                    'upper_bound', 'Gap(upper)', 'time(upper)', 
                                    'time (all)'
                                    ]
                                result_table = pd.DataFrame(index=index, columns=columns)
                                result_table.index.name = 'seed'
                                # write result_table
                                for demands_seed in tqdm.tqdm(demands_seeds_values, desc='demands_seed'.ljust(20), leave=False):
                                    # start!
                                    start_time = time.time()
                                    # set parameters
                                    params = Parameter(
                                        network_name=network_name, 
                                        graph=graph, 
                                        num_slots=num_slots, 
                                        num_demands=num_demands, 
                                        demands_population=demands_population, 
                                        demands_seed=demands_seed, 
                                        paths_dir=PATHS_FILE, 
                                        result_dir=RESULT_DIR, 
                                        bound_algo=bound_algo, 
                                        timelimit=timelimit, 
                                        width=width, 
                                        TRAFFIC_BPSK=TRAFFIC_BPSK
                                        )
                                    # optimize
                                    optimizer = PathChannelOptimizer(params)
                                    main_model_output, lower_bound_output, upper_bound_output = optimizer.run()
                                    # end!
                                    calc_time = time.time() - start_time
                                    # write result to result_table
                                    result_table.loc[demands_seed, 'lower_bound']   = int(lower_bound_output.lower_bound)
                                    result_table.loc[demands_seed, 'Gap(lower)']    = round(lower_bound_output.gap * 100, 2)
                                    result_table.loc[demands_seed, 'time(lower)']   = round(lower_bound_output.calculation_time, 3)
                                    # write upper bound result to result_table
                                    result_table.loc[demands_seed, 'upper_bound']   = int(upper_bound_output.upper_bound)
                                    result_table.loc[demands_seed, 'Gap(upper)']    = round(upper_bound_output.gap * 100, 2)
                                    result_table.loc[demands_seed, 'time(upper)']   = round(upper_bound_output.calculation_time, 3)
                                    # write main model result to result_table
                                    result_table.loc[demands_seed, 'used_slots']    = int(main_model_output.used_slots)
                                    result_table.loc[demands_seed, 'Gap(main)']     = round(main_model_output.gap * 100, 2)
                                    result_table.loc[demands_seed, 'time(main)']    = round(main_model_output.calculation_time, 3)
                                    # write whole time to result_table
                                    result_table.loc[demands_seed, 'time(all)']     = round(calc_time, 3)
                                    # save result_table
                                    result_table.to_csv(RESULT_FILE, index=True)
