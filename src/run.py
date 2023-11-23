import pickle
import pandas as pd

from src.utils.paths import RESULT_DIR
from src.optimize.params import Parameter
from src.optimize.optimizer import Optimizer
from src.utils.graph import create_network

if __name__ == "__main__":
    model_name = 'RSA_PATH_CHANNEL'
    network_name = 'NSF'
    graph = create_network(network_name)
    num_slots = 50
    num_demands = 20
    k_values = [2, 3, 5]
    algo_and_alpha = [('kSP', None), ('kBP', 0.3), ('kBP', 0.5)]
    # Test
    num_demands = 6
    experiment_num = str(input('Input experiment number: '))
    
    global_config = f"algorithm: {model_name}\nnetwork_name: {network_name}\nnum_slots: {num_slots}\nnum_demands: {num_demands}"
    with open(
        RESULT_DIR / f'experiment{experiment_num}/global_config.txt', 'w'
        ) as f:
        f.write(global_config)

    # run
    metrics = ['used_slots', 'calculation_time']
    algo_columns = [f'{algo}_{alpha}' for algo, alpha in algo_and_alpha]
    # metrics-algo_columnsのmulti-columnを作成
    columns = pd.MultiIndex.from_product([metrics, algo_columns])
    result_table = pd.DataFrame(index=k_values, columns=columns)
    for k in k_values:
        for path_algo_name, alpha in algo_and_alpha:
            sum_used_slots, sum_time = 0, 0
            for seed in range(1, 11):
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
                sum_used_slots += result['OptResult'].used_slots
                sum_time += result['OptResult'].calculation_time
                # save pickle
                file_name = f'k={k}_path_algo={path_algo_name}_alpha={alpha}_seed={seed}.pickle'
                # with open(
                #     RESULT_DIR / f'experiment{experiment_num}/raw_data' / file_name, 'wb'
                #     ) as f:
                #     pickle.dump(result, f)

            # average results
            avg_used_slots = sum_used_slots / 10
            avg_time = sum_time / 10
            # save results
            result_table.loc[k, ('used_slots', 
                                 f'{path_algo_name}_{alpha}')] = avg_used_slots
            result_table.loc[k, ('calculation_time',
                                f'{path_algo_name}_{alpha}')] = avg_time

    # save result_table
    result_table.to_csv(
        RESULT_DIR / f'experiment{experiment_num}/result_table.csv'
        )
                
