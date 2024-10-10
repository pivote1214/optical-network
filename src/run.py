import os
import sys

sys.path.append(os.path.abspath('..'))

import time
import argparse
# import logging
from tqdm import tqdm

import pandas as pd
import gurobipy as gp

from utils.network import load_network
from utils.namespaces import OUT_DIR
from utils.files import set_paths_file_path, set_result_dir
from src.optimize.models.RSA_PATH_CHANNEL.params import Parameter, TimeLimit, Width
from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer
from src.paths.algorithms.k_shortest_paths import KShortestPathsParams
from src.paths.algorithms.overall_optimization import NodePairClusteringParams
from src.paths.algorithms.k_balanced_paths import KSPwithSimilarityConstraintParams
from src.paths.algorithms.hierarchical_clustering import HierarchicalClusteringParams


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run optimization experiments.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    return parser.parse_args()

def write_global_config(ex_dir, config):
    config_file = os.path.join(ex_dir, 'global_config.txt')
    with open(config_file, 'w') as f:
        f.write('Global Configurations:\n')
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
    # logger.info(f'Global configuration written to {config_file}')

def initialize_parameters():
    return {
        'model_name': 'RSA_PATH_CHANNEL',
        'num_slots': 320,
        'num_demands': 100,
        'demands_population': [50, 100, 150, 200],
        'demands_seeds': [seed * 2 for seed in range(1, 11)],
        'network_names': ['JPN12', 'GRID3x4', 'NSF'],
        'path_algorithms': [
            'NodePairClustering', 
            'KShortestPaths', 
            'KSPwithSimilarityConstraint', 
            'HierarchicalClustering'
            ],
        'path_weights': ['hop'],
        'sim_weights': ['physical-length'],
        'cls_distances': ['single', 'average'],
        'alpha_list': [0.25, 0.50],
        'n_paths_list': [2, 3],
        'bound_algo': 'hybrid',
        'timelimit': TimeLimit(lower=30.0, upper=150.0, main=720.0),
        'width': Width(OC=37.5, GB=6.25, FS=12.5),
        'TRAFFIC_BPSK': 50,
        'threshold_values': [0.75],
        'w_obj_values': [0.5],
    }


def run_experiment(params, ex_dir):
    os.makedirs(ex_dir, exist_ok=True)
    write_global_config(ex_dir, {
        'num_slots': params['num_slots'],
        'num_demands': params['num_demands'],
        'demands_population': params['demands_population'],
        'bound_algo': params['bound_algo'],
    })
    
    algo_params_dict = {}

    for path_algo in params['path_algorithms']:
        algo_params_dict[path_algo] = set()
        for length_metric in params['path_weights']:
            for sim_metric in params['sim_weights']:
                for alpha in params['alpha_list']:
                    for threshold in params['threshold_values']:
                        for w_obj in params['w_obj_values']:
                            algo_params = create_algo_params(
                                path_algo, length_metric, sim_metric, alpha, threshold, w_obj
                            )
                            if algo_params is not None:
                                algo_params_dict[path_algo].add(algo_params)
                                        
    for network_name in tqdm(params['network_names'], desc="Networks".ljust(15)):
        graph = load_network(network_name)
        for n_paths in tqdm(params['n_paths_list'], desc="Number of Paths".ljust(15), leave=False):
            for path_algo, params_set in tqdm(algo_params_dict.items(), desc="Path Algorithms".ljust(15), leave=False):
                for algo_params in tqdm(params_set, desc="Params".ljust(15), leave=False):
                    run_rsa_for_algo(
                        params, network_name, graph, n_paths, path_algo, algo_params
                        )


def create_algo_params(path_algo, length_metric, sim_metric, alpha, threshold, w_obj):
    if path_algo == 'KShortestPaths':
        return KShortestPathsParams(length_metric=length_metric)
    elif path_algo == 'NodePairClustering':
        return NodePairClusteringParams(
            length_metric=length_metric,
            sim_metric=sim_metric,
            n_ref_paths=1,
            cutoff=None,
            linkage_method='average',
            criterion='distance',
            threshold=threshold,
            w_obj=w_obj,
            timelimit=600.0
        )
    elif path_algo == 'KSPwithSimilarityConstraint':
        return KSPwithSimilarityConstraintParams(
            length_metric=length_metric, 
            sim_metric=sim_metric, 
            alpha=alpha
        )
    elif path_algo == 'HierarchicalClustering':
        return HierarchicalClusteringParams(
            length_metric=length_metric, 
            sim_metric=sim_metric, 
            linkage_method='average'
        )
    else:
        # logger.warning(f'Path algorithm {path_algo} is not supported.')
        return None


def run_rsa_for_algo(params, network_name, graph, n_paths, path_algo, algo_params):
    paths_file = set_paths_file_path(path_algo, network_name, algo_params, n_paths)
    result_dir = set_result_dir(
        experiment_name=params['experiment_name'],
        algorithm=path_algo,
        network_name=network_name,
        params=algo_params,
        n_paths=n_paths
    )
    result_file = os.path.join(result_dir, 'summary.csv')

    if os.path.exists(result_file):
        # logger.info(f'Result file {result_file} already exists. Skipping.')
        return
    else:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    result_table = initialize_result_table(params['demands_seeds'])
    
    for demands_seed in tqdm(params['demands_seeds'], desc="Demands Seeds".ljust(15), leave=False):
        start_time = time.time()
        optimizer_params = Parameter(
            network_name=network_name, 
            graph=graph,
            num_slots=params['num_slots'],
            num_demands=params['num_demands'],
            demands_population=params['demands_population'],
            demands_seed=demands_seed,
            paths_dir=paths_file,
            result_dir=result_dir,
            bound_algo=params['bound_algo'],
            timelimit=params['timelimit'],
            width=params['width'],
            TRAFFIC_BPSK=params['TRAFFIC_BPSK']
        )
        optimizer = PathChannelOptimizer(optimizer_params)
        main_output, lower_output, upper_output = optimizer.run()
        calc_time = time.time() - start_time

        update_result_table(
            result_table, demands_seed, main_output,
            lower_output, upper_output, calc_time
        )
        result_table.to_csv(result_file, index=True)
        # logger.info(f'Results for seed {demands_seed} saved to {result_file}')


def initialize_result_table(seeds):
    columns = [
        'used_slots', 'Gap(main)', 'time(main)',
        'lower_bound', 'Gap(lower)', 'time(lower)',
        'upper_bound', 'Gap(upper)', 'time(upper)',
        'time(all)'
    ]
    result_table = pd.DataFrame(index=seeds, columns=columns)
    result_table.index.name = 'seed'
    return result_table

def update_result_table(table, seed, main_output, lower_output, upper_output, total_time):
    table.at[seed, 'used_slots'] = int(main_output.used_slots)
    table.at[seed, 'Gap(main)'] = round(main_output.gap * 100, 2)
    table.at[seed, 'time(main)'] = round(main_output.calculation_time, 3)

    table.at[seed, 'lower_bound'] = int(lower_output.lower_bound)
    table.at[seed, 'Gap(lower)'] = round(lower_output.gap * 100, 2)
    table.at[seed, 'time(lower)'] = round(lower_output.calculation_time, 3)

    table.at[seed, 'upper_bound'] = int(upper_output.upper_bound)
    table.at[seed, 'Gap(upper)'] = round(upper_output.gap * 100, 2)
    table.at[seed, 'time(upper)'] = round(upper_output.calculation_time, 3)

    table.at[seed, 'time(all)'] = round(total_time, 3)

def main():
    # dummy
    gp.Model()
    args = parse_arguments()
    params = initialize_parameters()
    params['experiment_name'] = args.experiment_name
    ex_dir = os.path.join(OUT_DIR, args.experiment_name)
    run_experiment(params, ex_dir)

if __name__ == "__main__":
    main()
