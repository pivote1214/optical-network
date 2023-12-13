from typing import Dict, List, Tuple, Optional

import time
import gurobipy as gp
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import DATA_DIR
from src.optimize.result import OptResult


@dataclass(frozen=True)
class PathChannelInput:
    E:      Dict[int, Tuple[int, int]]
    S:      List[int]
    D:      Dict[int, Tuple[int, int, int]]
    P:      Dict[int, List[List[int]]]
    C:      Dict[Tuple[int, int], List[List[int]]]
    delta:  Dict[Tuple[int, int, int], int]
    gamma:  Dict[Tuple[int, int, int, int], int]


class Model:
    def __init__(self, index_set: IndexSet, constant: Constant):
        self.index_set = index_set
        self.constant = constant
        self.name = 'RSA_PATH_CHANNEL'
        self.result = None
        self.calculation_time = None
        self.objective = None

    def _set_model(self) -> gp.Model:
        problem = gp.Model(self.name)
        # set variables
        self.x, self.y_es, self.y_s = {}, {}, {}
        # set variables x
        for d_ind, _ in self.index_set.D.items():
            for p_ind, _ in enumerate(self.index_set.P[d_ind]):
                for c_ind, _ in enumerate(self.index_set.C[d_ind, p_ind]):
                    self.x[d_ind, p_ind, c_ind] = problem.addVar(
                        vtype=gp.GRB.BINARY, name=f'x_{d_ind}_{p_ind}_{c_ind}'
                    )
        # set variables y_es
        for e_ind, _ in self.index_set.E.items():
            for s_ind, _ in enumerate(self.index_set.S):
                self.y_es[e_ind, s_ind] = problem.addVar(
                    vtype=gp.GRB.BINARY, name=f'y_es_{e_ind}_{s_ind}'
                )
        # set variables y_s
        for s_ind, _ in enumerate(self.index_set.S):
            self.y_s[s_ind] = problem.addVar(
                vtype=gp.GRB.BINARY, name=f'y_s_{s_ind}'
            )
        # update variables
        problem.update()

        # set objective function
        problem.setObjective(
            gp.quicksum(self.y_s[s_ind] * s_ind for s_ind in self.index_set.S), gp.GRB.MINIMIZE
            # gp.quicksum(self.y_s[s_ind] for s_ind in self.index_set.S), gp.GRB.MINIMIZE
        )
        # update objective function
        problem.update()

        # set constraints
        # set nonoverlap constraint
        for d_ind, _ in self.index_set.D.items():
            problem.addConstr(
                gp.quicksum(
                    self.x[d_ind, p_ind, c_ind] 
                    for p_ind, _ in enumerate(self.index_set.P[d_ind]) 
                    for c_ind, _ in enumerate(self.index_set.C[d_ind, p_ind])
                ) 
                == 1
            )
        # set y_es constraint
        for e_ind, _ in self.index_set.E.items():
            for s_ind, _ in enumerate(self.index_set.S):
                problem.addConstr(
                    gp.quicksum(
                        self.x[d_ind, p_ind, c_ind] * \
                            self.constant.gamma[d_ind, p_ind, c_ind, s_ind] * \
                                self.constant.delta[e_ind, d_ind, p_ind]
                            for d_ind, _ in self.index_set.D.items() 
                            for p_ind, _ in enumerate(self.index_set.P[d_ind]) 
                            for c_ind, _ in enumerate(self.index_set.C[d_ind, p_ind])
                    ) 
                    <= self.y_es[e_ind, s_ind]
                )
        # set y_s constraint
        for s_ind, _ in enumerate(self.index_set.S):
            problem.addConstr(
                gp.quicksum(
                    self.y_es[e_ind, s_ind] 
                    for e_ind, _ in self.index_set.E.items()
                ) 
                <= len(self.index_set.E) * self.y_s[s_ind]
            )
        # update constraints
        problem.update()

        return problem

    def variables_to_value(self) -> None:
        self.x = {key: var.X for key, var in self.x.items()}
        self.y_es = {key: var.X for key, var in self.y_es.items()}
        self.y_s = {key: var.X for key, var in self.y_s.items()}

    def calculate_used_slots(self) -> int:
        used_slots = 0
        for s_ind, _ in enumerate(self.index_set.S):
            used_slots += self.y_s[s_ind].X
        return used_slots
    
    def write_lp(self, dir_path: Optional[Path] = DATA_DIR / 'output' / 'lpfile') -> None:
        self.problem.write_lp(str(dir_path / f'{self.name}.lp'))

    def solve(
        self, 
        TIMELIMIT: Optional[int] = 3600, 
        write_lp: bool = False
        ) -> None:
        # set model
        self.problem = self._set_model()
        # set time limit
        self.problem.setParam('OutputFlag', 0)
        self.problem.setParam('TIMELIMIT', TIMELIMIT)

        # solve
        start = time.time()
        self.problem.optimize()
        elapsed_time = time.time() - start
        # store result
        if self.problem.Status == gp.GRB.INFEASIBLE:
            self.objective = None
            self.used_slots = None
        else:
            self.objective = self.problem.ObjVal
            self.used_slots = self.calculate_used_slots()
            self.variables_to_value()
        self.result = OptResult(
            status=self.problem.Status, 
            calculation_time=elapsed_time, 
            objective=self.objective, 
            used_slots=self.used_slots, 
            variable={'x': self.x, 'y_es': self.y_es, 'y_s': self.y_s}
            )
        if write_lp:
            self.write_lp()
