import warnings

from src.paths.base import BasePathSelector

__all__ = ["register_path_selector", "list_path_selectors"]

_path_selector_entrypoints = {}


def register_path_selector(fn: BasePathSelector) -> BasePathSelector:
    path_selector_name = fn.__name__

    if path_selector_name in _path_selector_entrypoints:
        warnings.warn(f"Duplicate path selector {path_selector_name}", stacklevel=2)

    _path_selector_entrypoints[path_selector_name] = fn

    return fn


def path_selector_entrypoint(path_selector_name: str) -> BasePathSelector:
    available_path_selectors = list_path_selectors()
    if path_selector_name not in available_path_selectors:
        raise ValueError(
            f"Invalid path selector name: {path_selector_name}, available path selectors are {available_path_selectors}"
        )

    return _path_selector_entrypoints[path_selector_name]


def list_path_selectors() -> list[str]:
    path_selectors = list(_path_selector_entrypoints.keys())
    return sorted(path_selectors)
