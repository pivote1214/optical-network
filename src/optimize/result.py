from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gurobipy as gp

@dataclass(frozen=False)
class OptResult:
    status: int
    calculation_time: float
    objective: float
    used_slots: float
    variable: dict[str, dict[gp.Var, Any]]

    def to_dict(self):
        return {**self.__dict__, **self.data_param.__dict__}
