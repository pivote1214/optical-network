from typing import Optional

import time
import gurobipy as gp
from dataclasses import dataclass


@dataclass(frozen=True)
class PathChannelInput:
    E:              dict[int, tuple[int, int]]
    S:              list[int]
    D:              dict[int, tuple[int, int, int]]
    P:              dict[int, list[list[int]]]
    C:              dict[tuple[int, int], list[list[int]]]
    delta:          dict[tuple[int, int, int], int]
    gamma:          dict[tuple[int, int, int, int], int]
    lower_bound:    int = None
    TIMELIMIT:      int = 600.0


@dataclass(frozen=True)
class PathChannelOutput:
    calculation_time:   float
    objective:          Optional[float]
    used_slots:         Optional[int]
    gap:                Optional[float]
    x:                  dict[tuple[int, int, int], int]
    y_es:               dict[tuple[int, int], int]
    y_s:                dict[int, int]


class PathChannelVariable:
    def __init__(self, input: PathChannelInput, problem: gp.Model):
        self.input:     PathChannelInput                    = input
        self.problem:   gp.Model                            = problem
        self.x:         dict[tuple[int, int, int], gp.Var]  = {}
        self.y_es:      dict[tuple[int, int], gp.Var]       = {}
        self.y_s:       dict[int, gp.Var]                   = {}

        self._set_variable()

    def set_variable(self) -> None:
        return self.problem

    def _set_variable(self) -> None:
        # set variables x
        for d_ind, _ in self.input.D.items():
            for p_ind, _ in enumerate(self.input.P[d_ind]):
                for c_ind, _ in enumerate(self.input.C[d_ind, p_ind]):
                    self.x[d_ind, p_ind, c_ind] = self.problem.addVar(
                        vtype=gp.GRB.BINARY, name=f'x_{d_ind}_{p_ind}_{c_ind}'
                    )
        # set variables y_es
        for e_ind, _ in self.input.E.items():
            for s_ind, _ in enumerate(self.input.S):
                self.y_es[e_ind, s_ind] = self.problem.addVar(
                    vtype=gp.GRB.BINARY, name=f'y_es_{e_ind}_{s_ind}'
                )
        # set variables y_s
        for s_ind, _ in enumerate(self.input.S):
            self.y_s[s_ind] = self.problem.addVar(
                vtype=gp.GRB.BINARY, name=f'y_s_{s_ind}'
            )
        # update variables
        self.problem.update()

    def to_values(self) -> None:
        self.x      = {key: int(var.X) for key, var in self.x.items()}
        self.y_es   = {key: int(var.X) for key, var in self.y_es.items()}
        self.y_s    = {key: int(var.X) for key, var in self.y_s.items()}


class PathChannelObjectiveFunction:
    input:      PathChannelInput
    variable:   PathChannelVariable
    problem:    gp.Model

    def set_objective_function(self) -> None:
        self.problem.setObjective(
            gp.quicksum(self.variable.y_s[s_ind] for s_ind in self.input.S), 
            gp.GRB.MINIMIZE
        )
        self.problem.update()


class PathChannelConstraint:
    input:      PathChannelInput
    variable:   PathChannelVariable
    problem:    gp.Model

    def set_constraint(self) -> None:
        # set nonoverlap constraint
        for d_ind, _ in self.input.D.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.variable.x[d_ind, p_ind, c_ind] 
                    for p_ind, _ in enumerate(self.input.P[d_ind]) 
                    for c_ind, _ in enumerate(self.input.C[d_ind, p_ind])
                ) 
                == 1
            )
        # set y_es constraint
        for e_ind, _ in self.input.E.items():
            for s_ind, _ in enumerate(self.input.S):
                self.problem.addConstr(
                    gp.quicksum(
                        self.variable.x[d_ind, p_ind, c_ind] * \
                            self.input.gamma[d_ind, p_ind, c_ind, s_ind] * \
                                self.input.delta[e_ind, d_ind, p_ind]
                            for d_ind, _ in self.input.D.items() 
                            for p_ind, _ in enumerate(self.input.P[d_ind]) 
                            for c_ind, _ in enumerate(self.input.C[d_ind, p_ind])
                    ) 
                    <= self.variable.y_es[e_ind, s_ind]
                )
        # set y_s constraint
        for s_ind, _ in enumerate(self.input.S):
            self.problem.addConstr(
                gp.quicksum(
                    self.variable.y_es[e_ind, s_ind] 
                    for e_ind, _ in self.input.E.items()
                ) 
                <= len(self.input.E) * self.variable.y_s[s_ind]
            )
        # set lower bound constraint
        if self.input.lower_bound is not None:
            self.problem.addConstr(
                gp.quicksum(
                    self.variable.y_s[s_ind]
                    for s_ind, _ in enumerate(self.input.S)
                ) 
                >= self.input.lower_bound
            )
        # update constraints
        self.problem.update()


class PathChannelModel(PathChannelObjectiveFunction, PathChannelConstraint):
    input:      PathChannelInput
    variable:   PathChannelVariable
    problem:    gp.Model
    
    def __init__(self, input: PathChannelInput):
        self.input  = input
        self.name   = "RSA/Path/Channel"

    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)
        # self.problem.setParam(gp.GRB.Param.OutputFlag, 0)
        
        self.variable = PathChannelVariable(input=self.input, problem=self.problem)
        self.problem = self.variable.set_variable()
        
        self.set_objective_function()
        self.set_constraint()

    def calculate_used_slots(self) -> int:
        used_slots = 0
        for s_ind, _ in enumerate(self.input.S):
            used_slots += self.variable.y_s[s_ind]
        return used_slots

    def solve(self) -> PathChannelOutput:
        # set model
        self._set_problem()
        
        # start!
        start = time.time()
        # set time limit
        self.problem.setParam(gp.GRB.Param.TimeLimit, self.input.TIMELIMIT)
        self.problem.optimize()
        # end!
        calculation_time = time.time() - start
        
        # store result
        if self.problem.Status == gp.GRB.INFEASIBLE or self.problem.SolCount == 0:
            self.objective = len(self.input.S)
            self.used_slots = len(self.input.S)
            self.gap = (self.objective - self.input.lower_bound) / self.input.lower_bound * 100
        else:
            self.objective = self.problem.ObjVal
            self.variable.to_values()
            self.used_slots = self.calculate_used_slots()
            self.gap = self.problem.MIPGap

        # save result
        self.output = PathChannelOutput(
            calculation_time=calculation_time, 
            objective=self.objective, 
            used_slots=self.used_slots, 
            gap=self.gap, 
            x=self.variable.x, 
            y_es=self.variable.y_es, 
            y_s=self.variable.y_s
        )

        return self.output
