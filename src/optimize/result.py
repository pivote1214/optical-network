from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar
from gurobipy import Model, GRB, Var, quicksum


IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Variable = TypeVar("Variable")


@dataclass(frozen=False)
class OptResult:
    calculation_time: float
    objective: float
    used_slots: float
    problem: Optional[Model] = None
    index_set: Optional[IndexSet] = None
    constant: Optional[Constant] = None
    variable: Optional[Variable] = None

    def to_dict(self):
        return {**self.__dict__, **self.data_param.__dict__}
