import time
import gurobipy as gp
from dataclasses import dataclass


@dataclass(frozen=True)
class PathLowerBoundInput:
    E:                  dict[int, tuple[int, int]]
    S:                  list[int]
    D:                  dict[int, tuple[int, int, int]]
    P:                  dict[int, list[list[int]]]
    num_slots:          dict[tuple[int, int], int]
    delta:              dict[tuple[int, int, int], int]
    LOWER_TIMELIMIT:    float = 30.0


@dataclass(frozen=True)
class PathLowerBoundOutput:
    calculation_time:   float
    lower_bound:        int
    gap:                float
    x:                  dict[tuple[int, int], int]
    F_use:              int


class PathLowerBoundVariable:
    def __init__(self, input: PathLowerBoundInput, problem: gp.Model):
        self.input:     PathLowerBoundInput             = input
        self.problem:   gp.Model                        = problem
        self.x:         dict[tuple[int, int], gp.Var]   = {}
        self.F_use:     gp.Var                          = None

        self._set_variable()

    def set_variable(self) -> None:
        return self.problem

    def _set_variable(self) -> None:
        # set variables x
        for d_ind, _ in self.input.D.items():
            for p_ind, _ in enumerate(self.input.P[d_ind]):
                self.x[d_ind, p_ind] = self.problem.addVar(
                    vtype=gp.GRB.BINARY, name=f'x_{d_ind}_{p_ind}'
                )
        # set variables F_use
        self.F_use = self.problem.addVar(
            vtype=gp.GRB.INTEGER, name=f'F_use'
        )
        # update variables
        self.problem.update()

    def to_values(self) -> None:
        self.x = {key: int(var.X) for key, var in self.x.items()}
        self.F_use = int(self.F_use.X)


class PathLowerBoundObjectiveFunction:
    input:      PathLowerBoundInput
    variable:   PathLowerBoundVariable
    problem:    gp.Model

    def set_objective_function(self) -> None:
        self.problem.setObjective(self.variable.F_use, gp.GRB.MINIMIZE)
        # update objective function
        self.problem.update()


class PathLowerBoundConstraint:
    input:      PathLowerBoundInput
    variable:   PathLowerBoundVariable
    problem:    gp.Model

    def set_constraint(self) -> None:
        # set nonoverlap constraint
        for d_ind, _ in self.input.D.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.variable.x[d_ind, p_ind] 
                    for p_ind, _ in enumerate(self.input.P[d_ind])
                ) 
                == 1
            )
        # set F_use constraint
        for e_ind, _ in self.input.E.items():
            self.problem.addConstr(
                gp.quicksum(
                    self.input.delta[e_ind, d_ind, p_ind] * \
                        self.input.num_slots[d_ind, p_ind] * \
                            self.variable.x[d_ind, p_ind] 
                            for d_ind, _ in self.input.D.items() 
                            for p_ind, _ in enumerate(self.input.P[d_ind])
                )
                <= self.variable.F_use
            )
        # update constraints
        self.problem.update()
        

class PathLowerBoundModel(PathLowerBoundObjectiveFunction, PathLowerBoundConstraint):
    input:      PathLowerBoundInput
    variable:   PathLowerBoundVariable
    problem:    gp.Model

    def __init__(self, input: PathLowerBoundInput):
        self.input  = input
        self.name   = "PathLowerBound"

    def _set_problem(self) -> None:
        self.problem = gp.Model(self.name)

        self.variable = PathLowerBoundVariable(input=self.input, problem=self.problem)
        self.problem = self.variable.set_variable()

        self.set_objective_function()
        self.set_constraint()

    def solve(self) -> PathLowerBoundOutput:
        self._set_problem()
        
        # set time limit
        self.problem.setParam(gp.GRB.Param.TimeLimit, self.input.LOWER_TIMELIMIT)
        # start!
        start = time.time()
        # optimize
        self.problem.optimize()
        # end!
        caculation_time = time.time() - start

        self.variable.to_values()
        # get lower bound
        if self.problem.SolCount == 0:
            lower_bound = None
            gap = None
        else:
            lower_bound = self.variable.F_use
            gap = self.problem.MIPGap
            x = self.variable.x

        # save result
        self.output = PathLowerBoundOutput(
            calculation_time=caculation_time,
            lower_bound=lower_bound, 
            gap=gap, 
            x=x, 
            F_use=self.variable.F_use
        )

        return self.output
