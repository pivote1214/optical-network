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
from src.optimize.models.RSA_PATH_CHANNEL.params import Parameter, TimeLimit, Width
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
    num_demands             = 100
    demands_population      = [50, 100, 150, 200]
    demands_seed            = 2
    # set parameters (Path-parameter)
    network_name            = 'NSF'
    path_algo               = 'k-shortest-paths'
    path_weight             = 'hop'
    n_paths                 = 3
    bound_algo              = "only"
    timelimit               = TimeLimit(lower=30.0, upper=150.0, main=720.0)
    width                   = Width(OC=37.5, GB=6.25, FS=12.5)
    TRAFFIC_BPSK            = 50

    # make graph
    graph = load_network(network_name)

    # run
    # set file path
    params = {
        'path_weight': path_weight, 
        }
    PATHS_FILE = set_paths_file_path(
        algorithm=path_algo, 
        network_name=network_name, 
        params=params, 
        n_paths=n_paths
        )
    for demands_seed in range(2, 21, 2):
        # set parameters
        params = Parameter(
            network_name=network_name, 
            graph=graph, 
            num_slots=num_slots, 
            num_demands=num_demands, 
            demands_population=demands_population, 
            demands_seed=demands_seed, 
            paths_dir=PATHS_FILE, 
            result_dir='.', 
            bound_algo=bound_algo, 
            timelimit=timelimit, 
            width=width, 
            TRAFFIC_BPSK=TRAFFIC_BPSK
            )
        # optimize
        optimizer = PathChannelOptimizer(params)
        lower_bound_output, upper_bound_output = optimizer.run()
        # from pprint import pprint
        # # logをファイルに出力
        # import pickle
        # # with open(PATHS_FILE, "rb") as f:
        # #     paths = pickle.load(f)
        # # with open("log_my.txt", "a") as f:
        # #     # パスファイルを読み込み
        # #     f.write("Paths:\n")
        # #     pprint(paths, stream=f)
        # #     f.write("Obj_val:\n")
        # #     f.write(str(lower_bound_output.lower_bound))
        print(lower_bound_output.lower_bound)
