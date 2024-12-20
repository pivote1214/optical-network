import json
import os
import time
from typing import Any
import random

import hydra
import networkx as nx
import pandas as pd
from gurobipy import setParam
from hydra.core.hydra_config import HydraConfig

from src.config import OfflineConfig
from src.graph import load_network
from src.paths import create_path_selector
from src.paths._utils import product_grid
from src.rsa.optical_network import Width
from src.rsa.path_channel import *


def pause_sync():
    """Google Driveを終了（同期停止）"""
    os.system('pkill "Google Drive"')
    print("Google Driveの同期を停止しました。")

def resume_sync():
    """Google Driveを起動（同期再開）"""
    os.system('open -a "Google Drive"')
    print("Google Driveの同期を再開しました。")


def set_result_dir(run_dir: str, **kwargs: Any) -> str:
    params_name = [
        f"{key}={str(value).replace('.', 'd')}" 
        for key, value in kwargs.items()
        ]
    params_name.sort()
    paths_dir = os.path.join(run_dir, *params_name)
    return paths_dir


def search_optimum_params(
    cfg: OfflineConfig, 
    graph: nx.DiGraph, 
    max_length: int, 
    run_dir: str, 
    ) -> dict[tuple[Any, Any], list[list[Any]]]:
    """
    最適なパラメータを探索
    """
    params = cfg.selector.params
    best_lb = float("inf")
    best_params = []
    for params in product_grid(params):
        path_selector = create_path_selector(
            cfg.selector.name, 
            graph, 
            cfg.selector.n_paths, 
            max_length=max_length, 
            **params
            )
        all_paths = path_selector.select_all_paths()
        lb_total = 0
        for seed in [cfg.demand.seed_lb * i for i in range(1, 101)]:
            lb = calc_lower_bound(cfg, graph, run_dir, seed, all_paths)
            lb_total += lb
        lb_ave = lb_total / 100
        if lb_ave < best_lb:
            best_lb = lb_ave
            best_params = [params]
        elif lb_ave == best_lb:
            best_params.append(params)

    best_params_dict = {idx: params for idx, params in enumerate(best_params)}
    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump(best_params_dict, f)

    return best_params


def calc_lower_bound(
    cfg: OfflineConfig, 
    graph: nx.DiGraph, 
    run_dir: str, 
    seed: int, 
    all_paths: dict[tuple[Any, Any], list[list[Any]]]
    ) -> None:
    """
    下界の計算
    """
    optimizer_params = Parameter(
        network_name=cfg.optical_network.name, 
        graph=graph, 
        num_slots=cfg.optical_network.num_slots, 
        num_demands=cfg.demand.number, 
        demands_population=cfg.demand.population, 
        demands_seed=seed, 
        all_paths=all_paths, 
        result_dir=run_dir, 
        bound_algo="lower only", 
        timelimit=TimeLimit(
            lower=cfg.optimizer.timelimit.lower, 
            upper=cfg.optimizer.timelimit.upper, 
            main=cfg.optimizer.timelimit.main
            ), 
        width=Width(
            optical_carrier=cfg.optical_network.width.optical_carrier, 
            guard_band=cfg.optical_network.width.guard_band, 
            frequency_slot=cfg.optical_network.width.frequency_slot
            ), 
        TRAFFIC_BPSK=cfg.optical_network.t_bpsk, 
        )
    optimizer = PathChannelOptimizer(optimizer_params)
    lb_output: PathLowerBoundOutput = optimizer.run()
    return lb_output.lower_bound


def run_rsa(
    cfg: OfflineConfig, 
    graph: nx.DiGraph, 
    max_length: int, 
    run_dir: str, 
    best_params: list[dict[str, Any]]
    ) -> None:
    """
    RSAの実行
    """
    params = random.choice(best_params)
    result_dir = set_result_dir(run_dir, **params)
    os.makedirs(result_dir, exist_ok=True)
    selector = create_path_selector(
        cfg.selector.name, 
        graph, 
        cfg.selector.n_paths, 
        max_length=max_length, 
        **params
        )
    all_paths = selector.select_all_paths()
    result_table = initialize_result_table()
    # n_nodes = int(graph.number_of_nodes())
    # num_demands = int(n_nodes * (n_nodes - 1) // 2)
    for seed in [cfg.demand.seed_rsa * i for i in range(1, 11)]:
        optimizer_params = Parameter(
            network_name=cfg.optical_network.name, 
            graph=graph, 
            num_slots=cfg.optical_network.num_slots, 
            num_demands=cfg.demand.number, 
            demands_population=cfg.demand.population, 
            demands_seed=seed, 
            all_paths=all_paths, 
            result_dir=result_dir, 
            bound_algo="with", 
            timelimit=TimeLimit(
                lower=cfg.optimizer.timelimit.lower, 
                upper=cfg.optimizer.timelimit.upper, 
                main=cfg.optimizer.timelimit.main
                ), 
            width=Width(
                optical_carrier=cfg.optical_network.width.optical_carrier, 
                guard_band=cfg.optical_network.width.guard_band, 
                frequency_slot=cfg.optical_network.width.frequency_slot
                ), 
            TRAFFIC_BPSK=cfg.optical_network.t_bpsk, 
            )
        optimizer = PathChannelOptimizer(optimizer_params)
        start = time.time()
        main_output, lb_output, ub_output = optimizer.run()
        total_time = time.time() - start
        # save results
        result_table = update_result_table(
            result_table, 
            seed, 
            main_output, 
            lb_output, 
            ub_output, 
            total_time
            )
        # save result table
        result_table.to_csv(os.path.join(result_dir, "result_table.csv"))


@hydra.main(config_path="config", config_name="main_offline", version_base="1.3")
def main(cfg: OfflineConfig) -> None:
    # convert to usage format
    graph = load_network(cfg.optical_network.name)
    max_length = max(mf[0] for mf in cfg.optical_network.modulation_formats)
    run_dir = os.path.abspath(HydraConfig.get().run.dir)
    # search optimum parameters
    best_params = search_optimum_params(cfg, graph, max_length, run_dir)
    # run RSA
    run_rsa(cfg, graph, max_length, run_dir, best_params)


def initialize_result_table() -> pd.DataFrame:
    """
    Initialize a result table DataFrame with predefined columns.

    Args:
        seeds (list): List of seed values for indexing.

    Returns:
        pd.DataFrame: Initialized result table.
    """
    columns = [
        "used_slots",
        "Gap(main)",
        "time(main)",
        "lower_bound",
        "Gap(lower)",
        "time(lower)",
        "upper_bound",
        "Gap(upper)",
        "time(upper)",
        "time(all)",
    ]
    result_table = pd.DataFrame(columns=columns)
    result_table.index.name = "seed"
    return result_table


def update_result_table(
    table: pd.DataFrame,
    seed: int,
    main_output: PathChannelOutput,
    lower_output: PathLowerBoundOutput,
    upper_output: PathUpperBoundOutput,
    total_time: float,
    ) -> pd.DataFrame:
    """
    Update the result table with the output of an optimization run.
    """
    table.at[seed, "used_slots"] = int(main_output.used_slots)
    table.at[seed, "Gap(main)"] = round(main_output.gap * 100, 2)
    table.at[seed, "time(main)"] = round(main_output.calculation_time, 3)

    table.at[seed, "lower_bound"] = int(lower_output.lower_bound)
    table.at[seed, "Gap(lower)"] = round(lower_output.gap * 100, 2)
    table.at[seed, "time(lower)"] = round(lower_output.calculation_time, 3)

    table.at[seed, "upper_bound"] = int(upper_output.upper_bound)
    table.at[seed, "Gap(upper)"] = round(upper_output.gap * 100, 2)
    table.at[seed, "time(upper)"] = round(upper_output.calculation_time, 3)

    table.at[seed, "time(all)"] = round(total_time, 3)

    return table


if __name__ == "__main__":
    setParam("LogToConsole", 0)
    pause_sync()
    main()
    resume_sync()
