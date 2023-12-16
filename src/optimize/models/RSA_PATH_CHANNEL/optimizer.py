from __future__ import annotations

from typing import Tuple, Optional

from src.optimize.params import Parameter
from src.optimize.models.RSA_PATH_CHANNEL.make_input import make_input_lower, make_channels, calculate_gamma
from src.optimize.models.RSA_PATH_CHANNEL.upper_bound import PathUpperBoundModel, PathUpperBoundInput, PathUpperBoundOutput
from src.optimize.models.RSA_PATH_CHANNEL.lower_bound import PathLowerBoundInput, PathLowerBoundModel, PathLowerBoundOutput
from src.optimize.models.RSA_PATH_CHANNEL.model import PathChannelModel, PathChannelInput, PathChannelOutput


class PathChannelOptimizer:
    def __init__(self, params: Parameter):
        self.params:    Parameter               = params
        self.input:     PathLowerBoundInput     = self._make_input()

    def run(self) -> Tuple[PathChannelOutput, Optional[PathLowerBoundOutput], Optional[PathUpperBoundOutput]]:
        if self.params.bound_algo:
            return self._run_with_bound_algo()
        else:
            return self._run_without_bound_algo()

    def _make_input(self) -> PathLowerBoundOutput:
        input = make_input_lower(self.params)
        return input

    def _run_with_bound_algo(self) -> Tuple[PathChannelOutput, PathLowerBoundOutput, PathUpperBoundOutput]:
        lower_bound_output = self._solve_lower_bound()
        upper_bound_output = self._solve_upper_bound()
        upper_bound = upper_bound_output.upper_bound + 1
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
            TIMELIMIT=self.params.TIMELIMIT
            )
        # run main model
        main_model = PathChannelModel(main_model_input)
        main_model_output = main_model.solve()

        return main_model_output, lower_bound_output, upper_bound_output

    def _run_without_bound_algo(self) -> Tuple[PathChannelOutput, None, None]:
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
            TIMELIMIT=self.params.TIMELIMIT
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
            M=320
            )
        upper_bound_model = PathUpperBoundModel(upper_bound_input)
        upper_bound_output = upper_bound_model.solve()
        return upper_bound_output
