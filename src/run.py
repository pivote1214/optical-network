import os
import pickle
from tkinter import TRUE
import gurobipy as gp
import pandas as pd

from src.utils.paths import RESULT_DIR
from src.optimize.params import Parameter
from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer

from src.utils.graph import load_network


if __name__ == "__main__":
    # dummy
    dummy = gp.Model('dummy')

    # experiment number
    experiment_num = str(input('Input experiment number: '))

    # make directory
    os.makedirs(RESULT_DIR / f'experiment{experiment_num}', exist_ok=True)
    for dir_name in ['main_model', 'lower_bound', 'upper_bound']:
        os.makedirs(RESULT_DIR / f'experiment{experiment_num}' / dir_name, exist_ok=True)

    # set parameters
    model_name              = 'RSA_PATH_CHANNEL'
    network_name            = 'EURO16'
    graph                   = load_network(network_name)
    num_slots               = 320
    num_demands             = 100
    demands_population      = [50, 100, 150, 200]
    demands_seeds_values    = [seed * 2 for seed in range(1, 11)]
    k_values                = [2]
    path_algo_infos         = [('kSP', None), ('kSPwLO', 0.3)]
    bound_algo              = True
    TIMELIMIT               = 3600.0

    # write global config
    with open(RESULT_DIR / f'experiment{experiment_num}/global_config.txt', 'w') as f:
        f.write(f'global config\n')
        f.write(f'model:                {model_name}\n')
        f.write(f'network_name:         {network_name}\n')
        f.write(f'num_slots:            {num_slots}\n')
        f.write(f'num_demands:          {num_demands}\n')
        f.write(f'demands_population:   {demands_population}\n')
        f.write(f'demands_seeds_values: {demands_seeds_values}\n')
        f.write(f'k_values:             {k_values}\n')
        f.write(f'path_algo_infos:      {path_algo_infos}\n')
        f.write(f'bound_algo:           {bound_algo}\n')
        f.write(f'TIMELIMIT:            {TIMELIMIT}\n')

    # run
    metrics = ['used_slots', 'Gap(main)', 'time(main)', 
               'lower_bound', 'Gap(lower)', 'time(lower)', 
               'upper_bound', 'Gap(upper)', 'time(upper)']
    algo_columns = [f'{algo}_{alpha}' for algo, alpha in path_algo_infos]
    # make multi-column
    columns = pd.MultiIndex.from_product([metrics, algo_columns])
    index = pd.MultiIndex.from_product([k_values, demands_seeds_values])
    result_table = pd.DataFrame(index=index, columns=columns)
    
    for k in k_values:
        for demands_seeds in demands_seeds_values:
            for path_algo_name, alpha in path_algo_infos:
                # set parameters
                params = Parameter(
                    network_name=network_name, 
                    graph=graph, 
                    num_slots=num_slots, 
                    num_demands=num_demands, 
                    demands_population=demands_population, 
                    demands_seed=demands_seeds, 
                    k=k, 
                    path_algo_name=path_algo_name, 
                    alpha=alpha, 
                    bound_algo=bound_algo, 
                    TIMELIMIT=TIMELIMIT
                    )

                # run
                optimizer = PathChannelOptimizer(params)
                main_model_output, lower_bound_output, upper_bound_output = optimizer.run()

                # set column name
                algo_column = f'{path_algo_name}_{alpha}'
                # write result to result_table
                if lower_bound_output is not None:
                    # write lower bound result to result_table
                    result_table.loc[(k, demands_seeds), 
                                    ('lower_bound', algo_column)] = \
                                        int(lower_bound_output.lower_bound)
                    result_table.loc[(k, demands_seeds), 
                                    ('Gap(lower)', algo_column)] = \
                                        round(lower_bound_output.gap * 100, 2)
                    result_table.loc[(k, demands_seeds), 
                                    ('time(lower)', algo_column)] = \
                                        round(lower_bound_output.calculation_time, 3)
                    # write upper bound result to result_table
                    result_table.loc[(k, demands_seeds), 
                                    ('upper_bound', algo_column)] = \
                                        int(upper_bound_output.upper_bound)
                    result_table.loc[(k, demands_seeds), 
                                    ('Gap(upper)', algo_column)] = \
                                        round(upper_bound_output.gap * 100, 2)
                    result_table.loc[(k, demands_seeds), 
                                    ('time(upper)', algo_column)] = \
                                        round(upper_bound_output.calculation_time, 3)
                # write main model result to result_table
                result_table.loc[(k, demands_seeds), 
                                ('used_slots', algo_column)] = \
                                    int(main_model_output.used_slots)
                result_table.loc[(k, demands_seeds), 
                                ('Gap(main)', algo_column)] = \
                                    round(main_model_output.gap * 100, 2)
                result_table.loc[(k, demands_seeds), 
                                ('time(main)', algo_column)] = \
                                    round(main_model_output.calculation_time, 3)
                # save result_table
                result_table.to_csv(RESULT_DIR / f'experiment{experiment_num}' / 'result_table.csv')

                # save outputs
                for dir_name in ['main_model', 'lower_bound', 'upper_bound']:
                    if dir_name == 'main_model':
                        output = main_model_output
                    elif dir_name == 'lower_bound':
                        output = lower_bound_output
                    elif dir_name == 'upper_bound':
                        output = upper_bound_output
                    if output is not None:
                        with open(RESULT_DIR / f'experiment{experiment_num}' / dir_name / \
                            f'k={k}_seeds={demands_seeds}_path={algo_column}_alpha={alpha}.pickle', 'wb') as f:
                            pickle.dump(output, f)
