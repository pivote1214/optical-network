import os
import pprint
import sys

sys.path.append(os.path.abspath('..'))


import gurobipy as gp

from utils.namespaces import TEST_DIR
from src.optimize.models.RSA_PATH_CHANNEL.lower_bound import PathLowerBoundOutput
from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer
from src.optimize.models.RSA_PATH_CHANNEL.params import Parameter, TimeLimit, Width
from utils.files import set_paths_file_path
from utils.network import load_network

def main(network_name: str, n_paths: int, n_demands: int):
    # dummy
    dummy = gp.Model('dummy')
    # set parameters (General)
    num_slots               = 320
    demands_population      = [50, 100, 150, 200]
    demands_seed            = 2
    # set parameters (Path-parameter)
    network_name            = network_name
    path_algo               = 'k-shortest-paths'
    path_weight             = 'hop'
    n_paths                 = 2
    bound_algo              = "lower only"
    timelimit               = TimeLimit(lower=30.0, upper=150.0, main=720.0)
    width                   = Width(OC=37.5, GB=6.25, FS=12.5)
    TRAFFIC_BPSK            = 50

    # make graph
    graph = load_network(network_name)

    # run
    params = {
        'path_weight': path_weight, 
        }
    PATHS_FILE = set_paths_file_path(
        algorithm=path_algo, 
        network_name=network_name, 
        params=params, 
        n_paths=n_paths
        )
    lower_bound_obj_vals = []
    for demands_seed in range(2, 21, 2):
        # set parameters
        params = Parameter(
            network_name=network_name, 
            graph=graph, 
            num_slots=num_slots, 
            num_demands=n_demands, 
            demands_population=demands_population, 
            demands_seed=demands_seed, 
            paths_dir=PATHS_FILE, 
            result_dir='./log', 
            bound_algo=bound_algo, 
            timelimit=timelimit, 
            width=width, 
            TRAFFIC_BPSK=TRAFFIC_BPSK
            )
        # optimize
        optimizer = PathChannelOptimizer(params)
        lower_bound_output: PathLowerBoundOutput = optimizer.run()
        lower_bound_obj_vals.append(lower_bound_output.lower_bound)


    print("Lower Bound Objective Values")
    pprint.pprint(lower_bound_obj_vals)


if __name__ == '__main__':
    main(
        network_name="GRID2x2",  
        n_paths=2, 
        n_demands=5
    )
