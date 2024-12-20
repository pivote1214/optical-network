import time
from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True)
class PathLowerBoundInput:
    E:                  dict[int, tuple[int, int]]
    S:                  list[int]
    D:                  dict[int, tuple[int, int, int]]
    P:                  dict[int, list[list[int]]]
    num_slots:          dict[tuple[int, int], int]
    delta:              dict[tuple[int, int, int], int]
    result_dir:         str
    demand_seed:        int
    timelimit:          float = 30.0


@dataclass(frozen=True)
class PathLowerBoundOutput:
    calculation_time:   float
    lower_bound:        int
    gap:                float
    x:                  dict[tuple[int, int], int]
    F_use:              int


class PathLowerBoundModel:
    def __init__(self, input: PathLowerBoundInput):
        self.input = input
        self.name = "PathLowerBound"
        self.problem = None
        self.x = {}
        self.F_use = None

    def _set_variables(self) -> None:
        # set variables x
        for d_ind, _ in self.input.D.items():
            for p_ind, _ in enumerate(self.input.P[d_ind]):
                self.x[d_ind, p_ind] = self.problem.addVar(
                    vtype=gp.GRB.BINARY, name=f'x_{d_ind}_{p_ind}'
                )
        # set variable F_use
        self.F_use = self.problem.addVar(
            vtype=gp.GRB.INTEGER, name='F_use'
        )
        # update variables
        self.problem.update()

    def _set_objective_function(self) -> None:
        self.problem.setObjective(self.F_use, gp.GRB.MINIMIZE)
        # update objective function
        self.problem.update()

    def _set_constraints(self) -> None:
        # set non-overlap constraint
        for d_ind, _ in self.input.D.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.x[d_ind, p_ind]
                    for p_ind, _ in enumerate(self.input.P[d_ind])
                ) == 1
            )
        # set F_use constraint
        for e_ind, _ in self.input.E.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.input.delta[e_ind, d_ind, p_ind] *
                    self.input.num_slots[d_ind, p_ind] *
                    self.x[d_ind, p_ind]
                    for d_ind, _ in self.input.D.items()
                    for p_ind, _ in enumerate(self.input.P[d_ind])
                ) <= self.F_use
            )
        # update constraints
        self.problem.update()

    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)
        self.problem.setParam('LogToConsole', 0)
        self.problem.setParam(gp.GRB.Param.Threads, 6)
        # set time limit
        self.problem.setParam(gp.GRB.Param.TimeLimit, self.input.timelimit)

        self._set_variables()
        self._set_objective_function()
        self._set_constraints()

    def solve(self) -> PathLowerBoundOutput:
        self._set_problem()

        # start!
        start = time.time()
        # optimize
        self.problem.optimize()
        # end!
        calculation_time = time.time() - start

        self._to_values()

        # get lower bound
        if self.problem.SolCount == 0:
            lower_bound = None
            gap = None
        else:
            lower_bound = self.F_use
            gap = self.problem.MIPGap

        # save result
        self.output = PathLowerBoundOutput(
            calculation_time=calculation_time,
            lower_bound=lower_bound,
            gap=gap,
            x=self.x,
            F_use=self.F_use
        )

        # self.ouput_info()

        return self.output

    def _to_values(self) -> None:
        self.x = {key: int(var.X) for key, var in self.x.items()}
        self.F_use = int(self.F_use.X)
