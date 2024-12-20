import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gurobipy as gp


@dataclass(frozen=True)
class PathChannelInput:
    E:              dict[int, tuple[int, int]]
    S:              list[int]
    D:              dict[int, tuple[int, int, int]]
    P:              dict[int, list[list[int]]]
    C:              dict[tuple[int, int], list[list[int]]]
    delta:          dict[tuple[int, int, int], bool]
    gamma:          dict[tuple[int, int, int, int], bool]
    lower_bound:    int
    result_dir:     str
    demand_seed:    int
    timelimit:      float = 600.0


@dataclass(frozen=True)
class PathChannelOutput:
    calculation_time:   float
    objective:          Optional[float]
    used_slots:         Optional[int]
    gap:                Optional[float]
    x:                  dict[tuple[int, int, int], int]
    y_es:               dict[tuple[int, int], int]
    y_s:                dict[int, int]


class PathChannelModel:
    def __init__(self, input: PathChannelInput):
        self.input = input
        self.name = "RSA/Path/Channel"
        self.problem = None
        self.x = {}
        self.y_es = {}
        self.y_s = {}

    def _set_variables(self) -> None:
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

    def _set_objective_function(self) -> None:
        self.problem.setObjective(
            gp.quicksum(self.y_s[s_ind] for s_ind in self.input.S), 
            gp.GRB.MINIMIZE
        )
        self.problem.update()

    def _set_constraints(self) -> None:
        # set nonoverlap constraint
        x_dind_indices = {
            d_ind: [
                (d_ind, p_ind, c_ind)
                for p_ind in range(len(self.input.P[d_ind]))
                for c_ind in range(len(self.input.C[d_ind, p_ind]))
            ]
            for d_ind in self.input.D
        }
        self.problem.addConstrs(
            (gp.quicksum(self.x[idx] for idx in x_dind_indices[d_ind]) == 1
            for d_ind in self.input.D),
            name='nonoverlap'
        )

        # set y_es constraint
        y_es_terms = defaultdict(list)

        # Build the mapping from (e_ind, s_ind) to list of (d_ind, p_ind, c_ind)
        for (d_ind, p_ind, c_ind, s_ind), gamma_val in self.input.gamma.items():
            if gamma_val:
                for e_ind in self.input.E:
                    if self.input.delta.get((e_ind, d_ind, p_ind), False):
                        y_es_terms[(e_ind, s_ind)].append((d_ind, p_ind, c_ind))

        for (e_ind, s_ind), indices in y_es_terms.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.x[d_ind, p_ind, c_ind] for (d_ind, p_ind, c_ind) in indices
                ) <= self.y_es[e_ind, s_ind]
            )


        # set y_s constraint
        y_es_eind_indices = {
            s_ind: [
                (e_ind, s_ind)
                for e_ind in self.input.E
            ]
            for s_ind in range(len(self.input.S))
        }
        self.problem.addConstrs(
            (gp.quicksum(self.y_es[idx] for idx in y_es_eind_indices[s_ind]) <= len(self.input.E) * self.y_s[s_ind]
            for s_ind in range(len(self.input.S))),
            name='y_s'
        )

        # set lower bound constraint
        if self.input.lower_bound is not None:
            self.problem.addConstr(
                gp.quicksum(self.y_s[s_ind] for s_ind in range(len(self.input.S))) >= self.input.lower_bound,
                name='lower_bound'
            )

        # No need to call update() when using addConstrs
        self.problem.update()


    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)
        # log to console: off, log file: on
        self.problem.setParam('LogToConsole', 0)
        self.problem.setParam(
            'LogFile', 
            os.path.join(self.input.result_dir, f'{self.input.demand_seed:02}_main.log')
            )
        # set time limit
        self.problem.setParam(gp.GRB.Param.TimeLimit, self.input.timelimit)
        # set thread
        self.problem.setParam(gp.GRB.Param.Threads, 6)

        self._set_variables()
        self._set_objective_function()
        self._set_constraints()

    def calculate_used_slots(self) -> int:
        used_slots = 0
        for s_ind, _ in enumerate(self.input.S):
            used_slots += self.y_s[s_ind]
        return used_slots

    def solve(self) -> PathChannelOutput:
        # set model
        self._set_problem()
        # start!
        start = time.time()
        # optimize
        self.problem.optimize()
        # end!
        calculation_time = time.time() - start

        self.problem.write(
            os.path.join(
                self.input.result_dir, 
                f'{self.input.demand_seed:02}_main.json'
                )
            )

        # store result
        if self.problem.Status == gp.GRB.INFEASIBLE or self.problem.SolCount == 0:
            self.objective = len(self.input.S)
            self.used_slots = len(self.input.S)
            self.gap = (self.objective - self.input.lower_bound) / self.input.lower_bound
        else:
            self.objective = self.problem.ObjVal
            self._to_values()
            self.used_slots = self.calculate_used_slots()
            self.gap = self.problem.MIPGap

        # save result
        self.output = PathChannelOutput(
            calculation_time=calculation_time,
            objective=self.objective,
            used_slots=self.used_slots,
            gap=self.gap,
            x=self.x,
            y_es=self.y_es,
            y_s=self.y_s
        )

        return self.output

    def _to_values(self) -> None:
        self.x = {key: int(var.X) for key, var in self.x.items()}
        self.y_es = {key: int(var.X) for key, var in self.y_es.items()}
        self.y_s = {key: int(var.X) for key, var in self.y_s.items()}
