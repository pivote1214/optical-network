from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Any
import gurobipy as gp

from src.optimize.params import Parameter
from src.utils.paths import OPT_MODEL_DIR
from src.utils.module_handler import get_object_from_module

IndexSet = TypeVar("IndexSet")
Constant = TypeVar("Constant")
Model = TypeVar("Model")


@dataclass
class ModelInput:
    model_name: str
    index_set: IndexSet
    constant: Constant


class Optimizer:
    def __init__(
        self,
        model_name: str, 
        params: Parameter
    ):
        self.model_name = model_name
        self.params = params
        self.result = {}

    @staticmethod
    def make_model_input(
        model_name: str, 
        params: Parameter
    ) -> ModelInput:
        """Generate model input"""
        module_path = OPT_MODEL_DIR / model_name / "make_input.py"
        make_input = get_object_from_module(module_path, f"make_input")
        index_set, constant = make_input(params=params)
        model_input = ModelInput(model_name=model_name, index_set=index_set, constant=constant)
        return model_input

    @staticmethod
    def make_model(model_input: ModelInput) -> gp.Model:
        """Bulid optimization model"""
        module_path = OPT_MODEL_DIR / model_input.model_name / "model.py"
        model_class = get_object_from_module(module_path, "Model")
        model = model_class(index_set=model_input.index_set, constant=model_input.constant)
        return model

    def run(self) -> dict[str, Any]:
        model_input = Optimizer.make_model_input(
            model_name=self.model_name, params=self.params
        )
        model = Optimizer.make_model(model_input)
        model.solve(self.params.TIMELIMIT)
        self.result['Parameters'] = self.params
        self.result['OptResult'] = model.result

        return self.result
