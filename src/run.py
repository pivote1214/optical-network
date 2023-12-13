import time
import pickle
from tqdm import tqdm
import gurobipy as gp
import pandas as pd

from src.utils.paths import RESULT_DIR
from src.optimize.params import Parameter
from src.optimize.optimizer import Optimizer
from src.utils.graph import load_network

if __name__ == "__main__":
    # dummy
    dummy = gp.Model('dummy')

    # set experiment number
    experiment_num = str(input('Input experiment number: '))

    # set parameters
    model_name      = 'RSA_PATH_CHANNEL'
    network_name    = 'NSF'
    graph           = load_network(network_name)
    num_slots       = 320
    num_demands     = 100
    k_values        = [2, 3, 5]
    algo_and_alpha  = [('kSP', None), ('kSPwLO', 0.3)]

    # write global config
    with open(RESULT_DIR / f'experiment{experiment_num}/global_config.txt', 'w') as f:
        f.write(f'global config\n')
        f.write(f'data: {time.ctime()}\n')
        f.write(f'algorithm: {model_name}\n')
        f.write(f'network_name: {network_name}\n')
        f.write(f'num_slots: {num_slots}\n')
        f.write(f'num_demands: {num_demands}\n')

    # run
    metrics = ['used_slots', 'calculation_time']
    algo_columns = ['seed'] + [f'{algo}_{alpha}' for algo, alpha in algo_and_alpha]
    # metrics-algo_columnsのmulti-columnを作成
    columns = pd.MultiIndex.from_product([metrics, algo_columns])
    index = pd.MultiIndex.from_product([k_values, [seed * 12 for seed in range(1, 11)]])
    result_table = pd.DataFrame(index=index, columns=columns)
    for k in tqdm(k_values):
        for seed in tqdm(range(1, 11), leave=False):
            for path_algo_name, alpha in tqdm(algo_and_alpha, leave=False):
                params = Parameter(
                    network_name=network_name, 
                    graph=graph, 
                    num_slots=num_slots, 
                    num_demands=num_demands, 
                    demands_seed=seed*12, 
                    k=k, 
                    path_algo_name=path_algo_name, 
                    alpha=alpha, 
                    TimeLimit=3600
                    )
                optimizer = Optimizer(model_name=model_name, params=params)
                result = optimizer.run()
                # record raw data
                result_table.loc[(k, seed*12), ('used_slots', 
                                 f'{path_algo_name}_{alpha}')] = result['OptResult'].used_slots
                result_table.loc[(k, seed*12), ('calculation_time',
                                f'{path_algo_name}_{alpha}')] = result['OptResult'].calculation_time

                # save result_table
                result_table.to_csv(
                    RESULT_DIR / f'experiment{experiment_num}/result_table.csv'
                    )
                # save pickle
                file_name = f'k={k}_path_algo={path_algo_name}_alpha={alpha}_seed={seed}.pickle'
                
                with open(
                    RESULT_DIR / f'experiment{experiment_num}/raw_data' / file_name, 'wb'
                    ) as f:
                    pickle.dump(result, f)

                # save result_table
                result_table.to_csv(
                    RESULT_DIR / f'experiment{experiment_num}/result_table.csv'
                    )
                
