from dataclasses import dataclass
from typing import Any

from src.rsa.optical_network import Width
from src.rsa.path_channel import TimeLimit

__all__ = [
    "OfflineConfig",
    "OnlineConfig",
    "OpticalNetworkConfig",
    "OfflineDemandConfig",
    "OptimizerConfig",
    "SelectorConfig",
]


@dataclass
class OpticalNetworkConfig:
    name: str
    num_slots: int
    t_bpsk: int
    width: Width
    modulation_formats: list[list[int, int]]

@dataclass
class OfflineDemandConfig:
    number: int
    population: list[int]
    seed_lb: int
    seed_rsa: int


@dataclass
class OnlineDemandConfig:
    number: int
    population: list[int]
    seed: int
    holding_time_ave: int
    erlangs: list[int]


@dataclass
class OptimizerConfig:
    name: str
    timelimit: TimeLimit


@dataclass
class SelectorConfig:
    name: str
    n_paths: int
    params: dict[str, list[Any]]


@dataclass
class OfflineConfig:
    exp_name: str
    optical_network: OpticalNetworkConfig
    demand: OfflineDemandConfig
    optimizer: OptimizerConfig
    selector: SelectorConfig


@dataclass
class OnlineConfig:
    exp_name: str
    optical_network: OpticalNetworkConfig
    demand: OnlineDemandConfig
    selector: SelectorConfig
