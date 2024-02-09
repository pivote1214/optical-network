import os
import time
import gurobipy as gp
import pandas as pd

from src.utils.paths import RESULT_DIR
from src.utils.graph import load_network
from src.optimize.params import Parameter
from src.optimize.models.RSA_PATH_CHANNEL.optimizer import PathChannelOptimizer


if __name__ == "__main__":
    # dummy
    dummy = gp.Model('dummy')

    # experiment number
    experiment_num = str(input('Input experiment number: '))

    # set parameters
    model_name              = 'RSA_PATH_CHANNEL'
    network_names           = ['NSF', 'JPN12', 'EURO16']
    num_slots               = 320
    num_demands             = 100
    demands_population      = [50, 100, 150, 200]
    demands_seeds_values    = [seed * 2 for seed in range(1, 11)]
    k_values                = [3]
    path_algo_infos         = [('kSP', None), ('kSP-hop', None), ('kSPwLO', 0.1), ('kSPwLO', 0.2), 
                               ('kSPwLO', 0.3), ('kSPwLO', 0.4), ('kSPwLO', 0.5), ('kSPwLO', 0.6), 
                               ('kSPwLO', 0.7), ('kSPwLO', 0.8), ('kSPwLO', 0.9), ('kDP', None)]
    bound_algo              = "hybrid"
    TIMELIMIT               = 600.0

    # make directory
    EX_DIR = RESULT_DIR / f'{model_name}/experiment_{experiment_num}'
    os.makedirs(EX_DIR, exist_ok=True)

    # write global config
    with open(RESULT_DIR / f'{model_name}/experiment_{experiment_num}/global_config.txt', 'w') as f:
        f.write(f'global config\n')
        f.write(f'num_slots:            {num_slots}\n')
        f.write(f'num_demands:          {num_demands}\n')
        f.write(f'demands_population:   {demands_population}\n')
        f.write(f'demands_seeds_values: {demands_seeds_values}\n')
        f.write(f'bound_algo:           {bound_algo}\n')
        f.write(f'TIMELIMIT:            {TIMELIMIT}\n')

    # run
    for network_name in network_names:
        EX_NET_DIR = EX_DIR / network_name
        os.makedirs(EX_NET_DIR, exist_ok=True)
        graph = load_network(network_name)
        for k in k_values:
            FILE_NAME = f'result_table_k={k}_sub.csv'
            # initialize result_table
            metrics = ['used_slots',  'Gap(main)',  'time(main)', 
                       'lower_bound', 'Gap(lower)', 'time(lower)', 
                       'upper_bound', 'Gap(upper)', 'time(upper)', 
                       'time (all)']
            algo_columns = [f'{algo}_{alpha}' for algo, alpha in path_algo_infos]
            # make multi-column
            columns = pd.MultiIndex.from_product([metrics, algo_columns])
            result_table = pd.DataFrame(index=demands_seeds_values, columns=columns)
            for demands_seeds in demands_seeds_values:
                for path_algo_name, alpha in path_algo_infos:
                    whole_start = time.time()
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
                    try:
                        # optimize
                        optimizer = PathChannelOptimizer(params)
                        main_model_output, lower_bound_output, upper_bound_output = optimizer.run()
                    except Exception as e:
                        print('Error:', e)
                        continue
                    whole_time = time.time() - whole_start

                    # set column name
                    algo_column = f'{path_algo_name}_{alpha}'
                    # write result to result_table
                    if lower_bound_output is not None:
                        # write lower bound result to result_table
                        result_table.loc[demands_seeds, 
                                        ('lower_bound', algo_column)] = \
                                            int(lower_bound_output.lower_bound)
                        result_table.loc[demands_seeds, 
                                        ('Gap(lower)', algo_column)] = \
                                            round(lower_bound_output.gap * 100, 2)
                        result_table.loc[demands_seeds, 
                                        ('time(lower)', algo_column)] = \
                                            round(lower_bound_output.calculation_time, 3)
                        # write upper bound result to result_table
                        result_table.loc[demands_seeds, 
                                        ('upper_bound', algo_column)] = \
                                            int(upper_bound_output.upper_bound)
                        result_table.loc[demands_seeds, 
                                        ('Gap(upper)', algo_column)] = \
                                            round(upper_bound_output.gap * 100, 2)
                        result_table.loc[demands_seeds, 
                                        ('time(upper)', algo_column)] = \
                                            round(upper_bound_output.calculation_time, 3)
                    # write main model result to result_table
                    result_table.loc[demands_seeds, 
                                    ('used_slots', algo_column)] = \
                                        int(main_model_output.used_slots)
                    result_table.loc[demands_seeds, 
                                    ('Gap(main)', algo_column)] = \
                                        round(main_model_output.gap * 100, 2)
                    result_table.loc[demands_seeds, 
                                    ('time(main)', algo_column)] = \
                                        round(main_model_output.calculation_time, 3)
                    # write whole time to result_table
                    result_table.loc[demands_seeds, 
                                    ('time (all)', algo_column)] = \
                                        round(whole_time, 3)
                    # save result_table
                    result_table.to_csv(EX_NET_DIR / FILE_NAME)
