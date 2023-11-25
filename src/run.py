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
    experiment_num = str(input('Input experiment number: '))
    model_name = 'RSA_PATH_CHANNEL'
    network_name = 'NSF'
    graph = load_network(network_name)
    num_slots = 75
    num_demands = 20
    k_values = [2, 3]
    algo_and_alpha = [('kSP', None), ('kSPwLO', 0.3), ('kSPwLO', 0.5)]
    # Test
    # num_slots = 30
    # num_demands = 6
    
    global_config = f"algorithm: {model_name}\nnetwork_name: {network_name}\nnum_slots: {num_slots}\nnum_demands: {num_demands}"
    with open(
        RESULT_DIR / f'experiment{experiment_num}/global_config.txt', 'w'
        ) as f:
        f.write(global_config)

    # run
    metrics = ['used_slots', 'calculation_time']
    algo_columns = ['seed'] + [f'{algo}_{alpha}' for algo, alpha in algo_and_alpha]
    # metrics-algo_columnsのmulti-columnを作成
    columns = pd.MultiIndex.from_product([metrics, algo_columns])
    index = pd.MultiIndex.from_product([k_values, [seed * 50 for seed in range(1, 11)]])
    result_table = pd.DataFrame(index=index, columns=columns)
    result_table_all = pd.DataFrame(index=k_values, columns=columns)
    for k in tqdm(k_values):
        for path_algo_name, alpha in tqdm(algo_and_alpha, leave=False):
            sum_used_slots, sum_time = 0, 0
            times = 10
            for seed in tqdm(range(1, 11), leave=False):
                params = Parameter(
                    network_name=network_name, 
                    graph=graph, 
                    num_slots=num_slots, 
                    num_demands=num_demands, 
                    demands_seed=seed*50, 
                    k=k, 
                    path_algo_name=path_algo_name, 
                    alpha=alpha, 
                    TimeLimit=3600
                    )
                optimizer = Optimizer(model_name=model_name, params=params)
                result = optimizer.run()
                # aggregate results
                if result['OptResult'].used_slots is None:
                    times -= 1
                    # print('infeasible')
                else:
                    sum_used_slots += result['OptResult'].used_slots
                    sum_time += result['OptResult'].calculation_time
                # save results
                result_table.loc[(k, seed*50), ('used_slots', 
                                 f'{path_algo_name}_{alpha}')] = result['OptResult'].used_slots
                result_table.loc[(k, seed*50), ('calculation_time',
                                f'{path_algo_name}_{alpha}')] = result['OptResult'].calculation_time

                # save result_table
                result_table.to_csv(
                    RESULT_DIR / f'experiment{experiment_num}/result_table.csv'
                    )
                # save pickle
                # file_name = f'k={k}_path_algo={path_algo_name}_alpha={alpha}_seed={seed}.pickle'
                # with open(
                #     RESULT_DIR / f'experiment{experiment_num}/raw_data' / file_name, 'wb'
                #     ) as f:
                #     pickle.dump(result, f)
                # # json
                # json_file_name = f'k={k}_path_algo={path_algo_name}_alpha={alpha}_seed={seed}.json'
                # with open(
                #     RESULT_DIR / f'experiment{experiment_num}/raw_data' / json_file_name, 'w'
                #     ) as f:
                #     json.dump(result, f, indent=4)

            # average results
            # print('times:', times)
            avg_used_slots = sum_used_slots / times
            avg_time = sum_time / times
            # save results
            result_table_all.loc[k, ('used_slots', 
                                 f'{path_algo_name}_{alpha}')] = avg_used_slots
            result_table_all.loc[k, ('calculation_time',
                                f'{path_algo_name}_{alpha}')] = avg_time
            # save result_table_all
            result_table_all.to_csv(
                RESULT_DIR / f'experiment{experiment_num}/result_table_all.csv'
                )
                
