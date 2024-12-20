import json
import pickle
import os
from typing import Any

import hydra
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

from src.config import OnlineConfig
from src.graph import load_network
from src.paths import create_path_selector
from src.paths._utils import product_grid
from src.rsa._utils import calc_bbr
from src.rsa.demands import gen_dynamic_demands
from src.rsa.online import simulate_online_rsa
from src.rsa.optical_network import Width


@hydra.main(config_path="config", config_name="main_online", version_base="1.3")
def main(cfg: OnlineConfig):
    # convert to usage format
    graph = load_network(cfg.optical_network.name)
    max_length = max(mf[0] for mf in cfg.optical_network.modulation_formats)
    run_dir = os.path.abspath(HydraConfig.get().run.dir)
    
    bbr_df = pd.DataFrame(columns=["bbr"])
    bbr_df.index.name = "erlang"
    test_erlang = cfg.demand.erlangs[4]
    best_params = get_best_params(cfg, graph, max_length, test_erlang)
    with open(os.path.join(run_dir, f"params_erlang={test_erlang}.json"), "w") as f:
        json.dump(best_params, f)
    for erlang in cfg.demand.erlangs:
        bbr, record = run_online_rsa(cfg, graph, max_length, erlang, best_params)
        bbr_df.loc[erlang] = bbr
        with open(os.path.join(run_dir, f"snapshot_erlang={erlang}.pkl"), "wb") as f:
            pickle.dump(record, f)
        
    bbr_df.to_csv(os.path.join(run_dir, "bbr.csv"))


def get_best_params(
    cfg: OnlineConfig, 
    graph: nx.Graph, 
    max_length: int, 
    erlang: int
    ) -> dict[str, Any]:
    params = cfg.selector.params
    best_bbr, best_params = np.inf, None
    for params in product_grid(params):
        # 候補パスの選択
        path_selector = create_path_selector(
            cfg.selector.name, 
            graph, 
            cfg.selector.n_paths, 
            max_length=max_length, 
            **params
            )
        all_paths = path_selector.select_all_paths()
        demands = gen_dynamic_demands(
            graph, 
            cfg.demand.number, 
            cfg.demand.population, 
            cfg.demand.holding_time_ave, 
            erlang, 
            cfg.demand.seed
            )
        allocated_demands, blocked_demands, _ = simulate_online_rsa(
            graph, 
            cfg.optical_network.num_slots, 
            demands, 
            all_paths, 
            cfg.optical_network.modulation_formats, 
            Width(
                optical_carrier=cfg.optical_network.width.optical_carrier, 
                guard_band=cfg.optical_network.width.guard_band, 
                frequency_slot=cfg.optical_network.width.frequency_slot
                ), 
            cfg.optical_network.t_bpsk
            )
        bbr = calc_bbr(allocated_demands, blocked_demands)
        if bbr < best_bbr:
            best_bbr = bbr
            best_params = params

    return best_params


def run_online_rsa(
    cfg: OnlineConfig, 
    graph: nx.Graph, 
    max_length: int, 
    erlang: int,
    best_params: dict[str, Any]
    ) -> tuple[float, list[tuple[Any, Any], NDArray[np.bool_]]]:
    # 候補パスの選択
    path_selector = create_path_selector(
        cfg.selector.name, 
        graph, 
        cfg.selector.n_paths, 
        max_length=max_length, 
        **best_params
        )
    all_paths = path_selector.select_all_paths()
    demands = gen_dynamic_demands(
        graph, 
        cfg.demand.number, 
        cfg.demand.population, 
        cfg.demand.holding_time_ave, 
        erlang, 
        cfg.demand.seed
        )
    allocated_demands, blocked_demands, record = simulate_online_rsa(
        graph, 
        cfg.optical_network.num_slots, 
        demands, 
        all_paths, 
        cfg.optical_network.modulation_formats, 
        Width(
            optical_carrier=cfg.optical_network.width.optical_carrier, 
            guard_band=cfg.optical_network.width.guard_band, 
            frequency_slot=cfg.optical_network.width.frequency_slot
            ), 
        cfg.optical_network.t_bpsk
        )
    bbr = calc_bbr(allocated_demands, blocked_demands)

    return bbr, record


if __name__ == "__main__":
    main()
