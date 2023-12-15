import os
import pprint
import gurobipy as gp
import pandas as pd

from src.utils.paths import RESULT_DIR
from src.optimize.params import Parameter
from src.optimize.models.RSA_PATH_CHANNEL import make_input
from src.optimize.models.RSA_PATH_CHANNEL.upper_bound import PathUpperBoundModel, PathUpperBoundInput
from src.optimize.models.RSA_PATH_CHANNEL.lower_bound import PathLowerBoundModel
from src.optimize.models.RSA_PATH_CHANNEL.model import PathChannelModel, PathChannelInput

from src.utils.graph import load_network

if __name__ == "__main__":
    # dummy
    dummy = gp.Model('dummy')

    # experiment number
    experiment_num = str(input('Input experiment number: '))

    # make directory
    os.makedirs(RESULT_DIR / f'experiment{experiment_num}', exist_ok=True)

    # set parameters
    model_name              = 'RSA_PATH_CHANNEL'
    network_name            = 'NSF'
    graph                   = load_network(network_name)
    num_slots               = 320
    num_demands             = 60
    demands_population      = [50, 100, 150, 200]
    demands_seeds_values    = [seed * 12 for seed in range(1, 11)]
    k_values                = [2, 3]
    path_algo_infos         = [('kSP', None), ('kSPwLO', 0.3)]
    bound_algo              = True
    TIMELIMIT               = 600

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
    metrics = ['used_slots', 'calculation_time', 'lower_bound', 'upper_bound']
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
                # run lower bound model
                lower_bound_input   = make_input.make_input_lower(params)
                lower_bound_model   = PathLowerBoundModel(lower_bound_input)
                lower_bound_output  = lower_bound_model.solve()
                
                # make upper bound input
                upper_bound_input = PathUpperBoundInput(
                    E=lower_bound_input.E, 
                    S=lower_bound_input.S, 
                    D=lower_bound_input.D, 
                    P=lower_bound_input.P, 
                    num_slots=lower_bound_input.num_slots, 
                    delta=lower_bound_input.delta, 
                    x=lower_bound_output.x, 
                    F_use=lower_bound_output.lower_bound, 
                    M=320
                    )
                # run upper bound model
                upper_bound_model   = PathUpperBoundModel(upper_bound_input)
                upper_bound_output  = upper_bound_model.solve()

                # make main model input
                upper_bound = upper_bound_output.upper_bound + 5
                S = list(range(upper_bound)) 
                C = make_input.make_channels(
                    S, lower_bound_input.num_slots
                    )
                gamma = make_input.calculate_gamma(
                    lower_bound_input.S, lower_bound_input.D, 
                    lower_bound_input.P, C
                    )
                main_model_input = PathChannelInput(
                    E=lower_bound_input.E, 
                    S=S, 
                    D=lower_bound_input.D, 
                    P=lower_bound_input.P, 
                    C=C, 
                    delta=lower_bound_input.delta, 
                    gamma=gamma, 
                    lower_bound=lower_bound_output.lower_bound, 
                    TIMELIMIT=TIMELIMIT
                    )
                # run main model
                main_model          = PathChannelModel(main_model_input)
                main_model_output   = main_model.solve()

                algo_column = f'{path_algo_name}_{alpha}'
                # write result to result_table
                result_table.loc[(k, demands_seeds), 
                                 ('used_slots', algo_column)] = \
                                     main_model_output.used_slots
                result_table.loc[(k, demands_seeds), 
                                 ('calculation_time', algo_column)] = \
                                     main_model_output.calculation_time
                result_table.loc[(k, demands_seeds), 
                                 ('lower_bound', algo_column)] = \
                                     lower_bound_output.lower_bound
                result_table.loc[(k, demands_seeds), 
                                 ('upper_bound', algo_column)] = \
                                     upper_bound_output.upper_bound

                # save result_table
                result_table.to_csv(RESULT_DIR / f'experiment{experiment_num}' / 'result_table.csv')
