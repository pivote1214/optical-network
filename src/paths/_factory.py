from typing import Any

import networkx as nx

from src.paths._registry import path_selector_entrypoint
from src.paths.base import BasePathSelector

__all__ = ["create_path_selector"]


def create_path_selector(
    path_selector_name: str,
    graph: nx.DiGraph,
    n_paths: int,
    max_length: int,
    **kwargs: Any,
    ) -> BasePathSelector:
    create_fn = path_selector_entrypoint(path_selector_name)
    return create_fn(graph, n_paths, max_length, **kwargs)
