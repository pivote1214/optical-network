import json

from src.optimize.params import Parameter
from src.optimize.optimizer import Optimizer
from src.utils.graph import create_network

if __name__ == "__main__":
    model_name = 'RSA_PATH_CHANNEL'
    network_name = 'NSF'
    graph = create_network(network_name)
    num_slots = 50
    num_demands = 10
    k_values = [2]
    algo_and_alpha = [('kSP', None), ('kBP', 0.3), ('kBP', 0.5)]

    for seed in range(1, 11):
        for k in k_values:
            for path_algo_name, alpha in algo_and_alpha:
                params = Parameter(
                    network_name=network_name, 
                    graph=graph, 
                    num_slots=num_slots, 
                    num_demands=num_demands, 
                    demands_seed=seed*100, 
                    k=k, 
                    path_algo_name=path_algo_name, 
                    alpha=alpha, 
                    TimeLimit=3600
                    )
                optimizer = Optimizer(model_name=model_name, params=params)
                result = optimizer.run()
                print(result['result'].objective)
                print(result['result'].used_slots)
