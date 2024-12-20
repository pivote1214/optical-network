from typing import Any

from src.rsa.path_channel import *
from src.rsa.path_channel.make_input import (
    calculate_gamma,
    make_channels,
    make_input_lower,
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

    def _run_only_bound_algo(
        self,
    ) -> tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_output = self._solve_upper_bound()

        return lower_bound_output, upper_bound_output

    def _run_with_bound_algo(
        self,
    ) -> tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_output = self._solve_upper_bound()
        upper_bound = int(upper_bound_output.upper_bound)
        lower_bound = int(lower_bound_output.lower_bound)
        if upper_bound == lower_bound:
            main_model_output = PathChannelOutput(
                calculation_time=0.0, 
                objective=upper_bound, 
                used_slots=upper_bound, 
                gap=0.0, 
                x=None, 
                y_es=None, 
                y_s=None
                )
        else:
            # make main model input
            S = list(range(upper_bound))
            C = make_channels(S, self.input.num_slots)
            gamma = calculate_gamma(self.input.S, self.input.D, self.input.P, C)
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
                timelimit=self.params.timelimit.main,
            )
            # run main model
            main_model = PathChannelModel(main_model_input)
            main_model_output = main_model.solve()

        return main_model_output, lower_bound_output, upper_bound_output

    def _run_without_bound_algo(self) -> tuple[PathChannelOutput, None, None]:
        # make main model input
        S = list(range(self.params.num_slots))
        C = make_channels(S, self.input.num_slots)
        gamma = calculate_gamma(self.input.S, self.input.D, self.input.P, C)
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
            timelimit=self.params.timelimit.main,
        )
        # run main model
        main_model = PathChannelModel(main_model_input)
        main_model_output = main_model.solve()

        return main_model_output, None, None

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
            timelimit=self.params.timelimit.upper,
        )
        upper_bound_model = PathUpperBoundModel(upper_bound_input)
        upper_bound_output = upper_bound_model.solve()
        return upper_bound_output
