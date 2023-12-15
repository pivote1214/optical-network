import pprint
from typing import Dict, List, Tuple

import time
import gurobipy as gp
from dataclasses import dataclass

from src.utils.graph import judge_common_edges


@dataclass(frozen=True)
class PathUpperBoundInput:
    E:          Dict[int, Tuple[int, int]]
    S:          List[int]
    D:          Dict[int, Tuple[int, int, int]]
    P:          Dict[int, List[List[int]]]
    num_slots:  Dict[Tuple[int, int], int]
    delta:      Dict[Tuple[int, int, int], int]
    x:          Dict[Tuple[int, int], int]
    F_use:      int
    M:          int


@dataclass(frozen=True)
class PathUpperBoundOutput:
    calculation_time:   float
    upper_bound:        int


class PathUpperBoundVariable:
    def __init__(self, input: PathUpperBoundInput, problem: gp.Model):
        self.input:     PathUpperBoundInput             = input
        self.problem:   gp.Model                        = problem
        self.o:         Dict[Tuple[int, int], gp.Var]   = {}
        self.f:         Dict[int, gp.Var]               = {}
        self.F_max:     gp.Var                          = None

        self._set_variable()

    def set_variable(self) -> None:
        return self.problem

    def _set_variable(self) -> None:
        # set variables o
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 != d_2:
                    self.o[d_1, d_2] = self.problem.addVar(
                        vtype=gp.GRB.BINARY, name=f'o_{d_1}_{d_2}'
                    )
        # set variables f
        for d_ind, _ in self.input.D.items():
            self.f[d_ind] = self.problem.addVar(
                vtype=gp.GRB.INTEGER, name=f'f_{d_ind}'
            )
        # set variables F_max
        self.F_max = self.problem.addVar(
            vtype=gp.GRB.INTEGER, name=f'F_max'
        )
        # update variables
        self.problem.update()

    def to_values(self) -> None:
        self.o = {key: var.X for key, var in self.o.items()}
        self.f = {key: var.X for key, var in self.f.items()}
        self.F_max = self.F_max.X


class PathUpperBoundObjectiveFunction:
    input:      PathUpperBoundInput
    variable:   PathUpperBoundVariable
    problem:    gp.Model

    def set_objective_function(self) -> None:
        self.problem.setObjective(self.variable.F_max, gp.GRB.MINIMIZE)
        # update objective function
        self.problem.update()


class PathUpperBoundConstraint:
    input:      PathUpperBoundInput
    variable:   PathUpperBoundVariable
    problem:    gp.Model

    def set_constraint(self) -> None:
        # set nonoverlap constraint
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 >= d_2:
                    continue
                self.problem.addConstr(
                    self.variable.o[d_1, d_2] + self.variable.o[d_2, d_1] == 1
                )
        # set slot index constraint
        for d_1, _ in self.input.D.items():
            for d_2, _ in self.input.D.items():
                if d_1 == d_2:
                    continue
                for p_1, _ in enumerate(self.input.P[d_1]):
                    for p_2, _ in enumerate(self.input.P[d_2]):
                        if judge_common_edges(self.input.P[d_1][p_1], self.input.P[d_2][p_2]):
                            self.problem.addConstr(
                                self.variable.f[d_1] + self.input.num_slots[d_1, p_1] 
                                <= self.variable.f[d_2] + \
                                    self.input.M * (3 - self.input.x[d_1, p_1] - self.input.x[d_2, p_2] - self.variable.o[d_1, d_2])
                                    )
        # define F_max
        for d_ind, _ in self.input.D.items():
            for p_ind, _ in enumerate(self.input.P[d_ind]):
                self.problem.addConstr(
                    self.variable.f[d_ind] + \
                        self.input.num_slots[d_ind, p_ind] * self.input.x[d_ind, p_ind] - 1
                    <= self.variable.F_max
                )
        # set F_max lower bound
        self.problem.addConstr(
            self.variable.F_max >= self.input.F_use
        )
        # update constraints
        self.problem.update()
        

class PathUpperBoundModel(PathUpperBoundObjectiveFunction, PathUpperBoundConstraint):
    input:      PathUpperBoundInput
    variable:   PathUpperBoundVariable
    problem:    gp.Model

    def __init__(self, input: PathUpperBoundInput):
        self.input  = input
        self.name   = "PathUpperBound"

    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)
        # self.problem.setParam(gp.GRB.Param.OutputFlag, False)
        self.problem.setParam(gp.GRB.Param.TimeLimit, 600)

        self.variable = PathUpperBoundVariable(input=self.input, problem=self.problem)
        self.problem = self.variable.set_variable()

        self.set_objective_function()
        self.set_constraint()

    def solve(self) -> PathUpperBoundOutput:
        self._set_problem()
        # start optimization
        start = time.time()
        self.problem.optimize()
        caculation_time = time.time() - start

        self.variable.to_values()
        upper_bound = int(self.variable.F_max)

        # save result
        self.output = PathUpperBoundOutput(
            calculation_time=caculation_time, 
            upper_bound=upper_bound
        )

        return self.output

        
