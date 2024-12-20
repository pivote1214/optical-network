from typing import Any, Iterator
from itertools import product

__all__ = ["product_grid"]


def product_grid(params: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """
    与えられたパラメータ候補の全組み合わせを生成
    """
    param_names = list(params.keys())
    param_values = list(params.values())
    for combination in product(*param_values):
        yield dict(zip(param_names, combination))
