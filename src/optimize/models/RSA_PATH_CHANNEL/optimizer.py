import os
import sys
sys.path.append(os.path.abspath('../../../../'))

import time
from typing import Any, Optional

from src.optimize.algorithms.greedy import greedy_RMLSA_offline
from src.optimize.models.RSA_PATH_CHANNEL.lower_bound import (
    PathLowerBoundInput,
    PathLowerBoundModel,
    PathLowerBoundOutput,
)
from src.optimize.models.RSA_PATH_CHANNEL.make_input import (
    calculate_gamma,
    make_channels,
    make_input_lower,
)
from src.optimize.models.RSA_PATH_CHANNEL.model import (
    PathChannelInput,
    PathChannelModel,
    PathChannelOutput,
)
from src.optimize.models.RSA_PATH_CHANNEL.params import Parameter
from src.optimize.models.RSA_PATH_CHANNEL.upper_bound import (
    PathUpperBoundInput,
    PathUpperBoundModel,
    PathUpperBoundOutput,
)


class PathChannelOptimizer:
    def __init__(self, params: Parameter):
        self.params: Parameter = params
        self.input: PathLowerBoundInput = self._make_input()


    def run(self) -> Any:
        if self.params.bound_algo == "with":
            return self._run_with_bound_algo()
        elif self.params.bound_algo == "only":
            return self._run_only_bound_algo()
        elif self.params.bound_algo == "hybrid":
            return self._run_hybrid()
        elif self.params.bound_algo == "lower only":
            return self._solve_lower_bound()
        elif self.params.bound_algo == "without":
            return self._run_without_bound_algo()

    def _make_input(self) -> PathLowerBoundInput:
        input = make_input_lower(self.params)
        return input

    def _run_only_bound_algo(self) -> tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_output = self._solve_upper_bound()

        return lower_bound_output, upper_bound_output
    
    def _run_with_bound_algo(self) -> tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_output = self._solve_upper_bound()
        upper_bound = int(upper_bound_output.upper_bound)
        # make main model input
        S = list(range(upper_bound))
        C = make_channels(
            S, self.input.num_slots
            )
        gamma = calculate_gamma(
            self.input.S, self.input.D, 
            self.input.P, C
            )
        main_model_input = PathChannelInput(
            E=self.input.E, 
            S=S, 
            D=self.input.D, 
            P=self.input.P, 
            C=C, 
            delta=self.input.delta, 
            gamma=gamma, 
            lower_bound=lower_bound_output.lower_bound, 
            result_dir=self.params.result_dir, 
            demand_seed=self.params.demands_seed, 
            timelimit=self.params.timelimit.main
            )
        # run main model
        main_model = PathChannelModel(main_model_input)
        main_model_output = main_model.solve()

        return main_model_output, lower_bound_output, upper_bound_output

    def _run_without_bound_algo(self) -> tuple[PathChannelOutput, None, None]:
        # make main model input
        S = list(range(self.params.num_slots))
        C = make_channels(
            S, self.input.num_slots
            )
        gamma = calculate_gamma(
            self.input.S, self.input.D, 
            self.input.P, C
            )
        main_model_input = PathChannelInput(
            E=self.input.E, 
            S=S, 
            D=self.input.D, 
            P=self.input.P, 
            C=C, 
            delta=self.input.delta, 
            gamma=gamma, 
            lower_bound=None, 
            result_dir=self.params.result_dir, 
            demand_seed=self.params.demands_seed, 
            timelimit=self.params.timelimit.main
            )
        # run main model
        main_model = PathChannelModel(main_model_input)
        main_model_output = main_model.solve()

        return main_model_output, None, None

    def _run_hybrid(self) -> tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        if lower_bound_output.lower_bound is None:
            lower_bound = None
            start = time.time()
            upper_bound = self._solve_first_fit()
            calculation_time = time.time() - start
            upper_bound_output = PathUpperBoundOutput(
                calculation_time=calculation_time, 
                upper_bound=upper_bound, 
                gap=None, 
                o={}, 
                f={}, 
                F_max=0
                )
        else:
            lower_bound = lower_bound_output.lower_bound
            upper_bound_output = self._solve_upper_bound()
            if upper_bound_output.upper_bound is None:
                upper_bound = self._solve_first_fit()
                upper_bound_output.upper_bound = upper_bound
            else:
                upper_bound = int(upper_bound_output.upper_bound)
        
        # make main model input
        S = list(range(upper_bound))
        C = make_channels(
            S, self.input.num_slots
            )
        gamma = calculate_gamma(
            self.input.S, self.input.D, 
            self.input.P, C
            )
        main_model_input = PathChannelInput(
            E=self.input.E, 
            S=S, 
            D=self.input.D, 
            P=self.input.P, 
            C=C, 
            delta=self.input.delta, 
            gamma=gamma, 
            lower_bound=lower_bound, 
            result_dir=self.params.result_dir, 
            demand_seed=self.params.demands_seed, 
            timelimit=self.params.timelimit.main
            )
        # run main model
        main_model = PathChannelModel(main_model_input)
        main_model_output = main_model.solve()
        if main_model_output.used_slots is None:
            main_model_output.used_slots = upper_bound

        return main_model_output, lower_bound_output, upper_bound_output

    def _solve_first_fit(self) -> PathChannelOutput:
        graph_name = self.params.network_name
        num_slots = self.params.num_slots
        k = self.params.k
        demands = self.input.D
        path_method = self.params.path_algo_name
        alpha = self.params.alpha
        upper_bound_ff = greedy_RMLSA_offline(
            graph_name, num_slots, k, demands, path_method, alpha
            )
        return upper_bound_ff

    def _solve_lower_bound(self) -> PathLowerBoundOutput:
        lower_bound_model = PathLowerBoundModel(self.input)
        lower_bound_output = lower_bound_model.solve()
        
        return lower_bound_output

    def _solve_upper_bound(self) -> PathUpperBoundOutput:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_input = PathUpperBoundInput(
            E=self.input.E, 
            S=self.input.S, 
            D=self.input.D, 
            P=self.input.P, 
            num_slots=self.input.num_slots, 
            delta=self.input.delta, 
            x=lower_bound_output.x, 
            F_use=lower_bound_output.lower_bound, 
            big_M=320, 
            result_dir=self.params.result_dir, 
            demand_seed=self.params.demands_seed, 
            timelimit=self.params.timelimit.upper
            )
        upper_bound_model = PathUpperBoundModel(upper_bound_input)
        upper_bound_output = upper_bound_model.solve()
        return upper_bound_output
