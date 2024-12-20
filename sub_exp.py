import os
import time
from typing import Any

import hydra
import networkx as nx
import pandas as pd
from gurobipy import setParam
from hydra.core.hydra_config import HydraConfig

from src.config import OfflineConfig
from src.graph import load_network
from src.paths import create_path_selector
from src.rsa.optical_network import Width
from src.rsa.path_channel import *


def set_result_dir(run_dir: str, **kwargs: Any) -> str:
    params_name = [
        f"{key}={str(value).replace('.', 'd')}" 
        for key, value in kwargs.items()
        ]
    params_name.sort()
    paths_dir = os.path.join(run_dir, *params_name)
    return paths_dir


def run_rsa(
    cfg: OfflineConfig, 
    graph: nx.DiGraph, 
    max_length: int, 
    run_dir: str, 
    ) -> None:
    """
    RSAの実行
    """
    result_dir = set_result_dir(run_dir)
    os.makedirs(result_dir, exist_ok=True)
    all_paths = set_all_paths(cfg, graph, max_length)
    result_table = initialize_result_table()
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


def set_all_paths(
    cfg: OfflineConfig, 
    graph: nx.DiGraph, 
    max_length: int
    ) -> dict[tuple[Any, Any], list[list[Any]]]:
    selector_name = cfg.selector.name
    # parameter
    length_metric = "hop"
    sim_metric = "physical-length"
    alpha = 0.5
    linkage_method = "average"
    n_ref_paths = 5
    threshold = 0.5
    beta = 0.5

    if selector_name == "ksp":
        selector = create_path_selector(
            selector_name,
            graph=graph,
            n_paths=cfg.selector.n_paths,
            max_length=max_length,
            length_metric=length_metric,
        )
    elif selector_name == "kspasc":
        selector = create_path_selector(
            selector_name,
            graph=graph,
            n_paths=cfg.selector.n_paths,
            max_length=max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            alpha=alpha,
        )
    elif selector_name == "kdp":
        selector = create_path_selector(
            selector_name,
            graph=graph,
            n_paths=cfg.selector.n_paths,
            max_length=max_length,
            sim_metric=sim_metric,
        )
    elif selector_name == "path_clst":
        selector = create_path_selector(
            selector_name,
            graph=graph,
            n_paths=cfg.selector.n_paths,
            max_length=max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            linkage_method=linkage_method,
        )
    elif selector_name == "nodepair_clst":
        selector = create_path_selector(
            selector_name,
            graph=graph,
            n_paths=cfg.selector.n_paths,
            max_length=max_length,
            length_metric=length_metric,
            sim_metric=sim_metric,
            linkage_method=linkage_method,
            n_ref_paths=n_ref_paths,
            threshold=threshold,
            beta=beta,
        )
    return selector.select_all_paths()


@hydra.main(config_path="config", config_name="main_offline", version_base="1.3")
def main(cfg: OfflineConfig) -> None:
    # convert to usage format
    graph = load_network(cfg.optical_network.name)
    max_length = max(mf[0] for mf in cfg.optical_network.modulation_formats)
    run_dir = os.path.abspath(HydraConfig.get().run.dir)
    # run RSA
    run_rsa(cfg, graph, max_length, run_dir)


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
    main()
