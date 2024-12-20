import os
import time
from dataclasses import dataclass

import gurobipy as gp

from src.graph import judge_common_edges


@dataclass(frozen=True)
class PathUpperBoundInput:
    E: dict[int, tuple[int, int]]
    S: list[int]
    D: dict[int, int, int]
    P: dict[int, list[list[int]]]
    num_slots: dict[tuple[int, int], int]
    delta: dict[tuple[int, int, int], int]
    x: dict[tuple[int, int], int]
    F_use: int
    big_M: int
    result_dir: str
    demand_seed: int
    timelimit: float = 150.0


@dataclass(frozen=True)
class PathUpperBoundOutput:
    calculation_time: float
    upper_bound: int
    gap: float
    o: dict[tuple[int, int], int]
    f: dict[int, int]
    F_max: int


class PathUpperBoundModel:
    def __init__(self, input: PathUpperBoundInput):
        self.input = input
        self.name = "PathUpperBound"
        self.problem = None
        self.o = {}
        self.f = {}
        self.F_max = None

    def _set_variables(self) -> None:
        # set variables o
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 != d_2:
                    self.o[d_1, d_2] = self.problem.addVar(
                        vtype=gp.GRB.BINARY, name=f"o_{d_1}_{d_2}"
                    )
        # set variables f
        for d_ind, _ in self.input.D.items():
            self.f[d_ind] = self.problem.addVar(vtype=gp.GRB.INTEGER, name=f"f_{d_ind}")
        # set variables F_max
        self.F_max = self.problem.addVar(vtype=gp.GRB.INTEGER, name="F_max")
        # update variables
        self.problem.update()

    def _set_objective_function(self) -> None:
        self.problem.setObjective(self.F_max, gp.GRB.MINIMIZE)
        # update objective function
        self.problem.update()

    def _set_constraints(self) -> None:
        # set non-overlap constraint
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 >= d_2:
                    continue
                self.problem.addConstr(self.o[d_1, d_2] + self.o[d_2, d_1] == 1)
        # set slot index constraint
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 == d_2:
                    continue
                for p_1, _ in enumerate(self.input.P[d_1]):
                    for p_2, _ in enumerate(self.input.P[d_2]):
                        if judge_common_edges(
                            self.input.P[d_1][p_1], self.input.P[d_2][p_2]
                        ):
                            self.problem.addConstr(
                                self.f[d_1] + self.input.num_slots[d_1, p_1]
                                <= self.f[d_2]
                                + self.input.big_M
                                * (
                                    3
                                    - self.input.x[d_1, p_1]
                                    - self.input.x[d_2, p_2]
                                    - self.o[d_1, d_2]
                                )
                            )
        # define F_max
        for d_ind, _ in self.input.D.items():
            for p_ind, _ in enumerate(self.input.P[d_ind]):
                self.problem.addConstr(
                    self.f[d_ind]
                    + self.input.num_slots[d_ind, p_ind] * self.input.x[d_ind, p_ind]
                    <= self.F_max
                )
        # set F_max lower bound
        self.problem.addConstr(self.F_max >= self.input.F_use)
        # update constraints
        self.problem.update()

    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)
        # log to console: off, log file: on
        self.problem.setParam("LogToConsole", 0)
        self.problem.setParam(
            'LogFile', 
            os.path.join(self.input.result_dir, f'{self.input.demand_seed:02}_upper.log')
            )
        # set time limit
        self.problem.setParam(gp.GRB.Param.TimeLimit, self.input.timelimit)
        # set thread
        self.problem.setParam(gp.GRB.Param.Threads, 6)

        self._set_variables()
        self._set_objective_function()
        self._set_constraints()

    def solve(self) -> PathUpperBoundOutput:
        self._set_problem()

        # start!
        start = time.time()
        # optimize
        self.problem.optimize()
        # no solution
        if self.problem.SolCount == 0:
            self.problem.setParam(gp.GRB.Param.TimeLimit, "inf")
            self.problem.optimize()
        # end!
        calculation_time = time.time() - start

        self.problem.write(
            os.path.join(
                self.input.result_dir, f"{self.input.demand_seed:02}_upper.json"
            )
        )

        # to values
        self._to_values()

        upper_bound = self.F_max

        # save result
        self.output = PathUpperBoundOutput(
            calculation_time=calculation_time,
            upper_bound=upper_bound,
            gap=self.problem.MIPGap,
            o=self.o,
            f=self.f,
            F_max=self.F_max,
        )

        return self.output

    def _to_values(self) -> None:
        self.o = {key: int(var.X) for key, var in self.o.items()}
        self.f = {key: int(var.X) for key, var in self.f.items()}
        self.F_max = int(self.F_max.X)
