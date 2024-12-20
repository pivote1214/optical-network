import json
import os
import time
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any

import networkx as nx

from src.files import load_pickle, save_pickle, set_paths_dir
from src.graph import calc_all_simple_paths
from src.namespaces import PATHS_DIR

__all__ = ["BasePathSelector", "BasePathSelectorSinglePair"]


class BasePathSelector(ABC):
    def __init__(
        self,
        graph: nx.DiGraph,
        n_paths: int,
        max_length: int,
        **kwargs: Any,
    ):
        self.graph = graph
        self.graph_name = graph.name
        self.n_paths = n_paths
        self.max_length = max_length
        self.paths_dir = set_paths_dir(
            path_selector=self.__class__.__name__,
            network_name=self.graph_name,
            n_paths=self.n_paths,
            **kwargs,
        )
        self.is_calculated = self.is_calculated()
        self.n_paths_per_pair = self.distribute_n_paths()
        self.all_simple_paths = self.set_all_simple_paths()

    def set_all_simple_paths(self) -> dict[tuple[Any, Any], list[list[Any]]]:
        """method to set all simple paths"""
        file_path = os.path.join(PATHS_DIR, self.graph_name, "all_simple_paths.pkl")
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            start = time.time()
            node_pairs = list(combinations(self.graph.nodes, 2))
            all_simple_paths = {
                (u, v): calc_all_simple_paths(self.graph, u, v, self.max_length) 
                for u, v in node_pairs
            }
            end = time.time()
            save_pickle(all_simple_paths, file_path)
            with open(os.path.join(os.path.dirname(file_path), "time.json"), "w") as jsonfile:
                json.dump({"time": end - start}, jsonfile)
        else:
            all_simple_paths = load_pickle(file_path)
        return all_simple_paths
        
    def is_calculated(self) -> bool:
        paths_file = os.path.join(self.paths_dir, "paths.pkl")
        if not os.path.exists(paths_file):
            os.makedirs(self.paths_dir, exist_ok=True)
            return False
        else:
            return True

    def select_all_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        """method to select all paths"""
        if not self.is_calculated:
            return self.save_selected_paths()
        else:
            return self._load_selected_paths()

    @abstractmethod
    def _select_all_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        """method to select all paths"""
        raise NotImplementedError

    def _load_selected_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        """method to load selected paths"""
        paths_file = os.path.join(self.paths_dir, "paths.pkl")
        return load_pickle(paths_file)

    def save_selected_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        """method to save selected paths"""
        start_time = time.time()
        all_paths = self._select_all_paths()
        end_time = time.time()
        # save paths
        paths_file = os.path.join(self.paths_dir, "paths.pkl")
        save_pickle(all_paths, paths_file)
        # save time to json file
        time_file = os.path.join(self.paths_dir, "time.json")
        with open(time_file, "w") as jsonfile:
            json.dump({"time": end_time - start_time}, jsonfile)
        return all_paths
        
    def distribute_n_paths(self) -> dict[tuple[Any, Any], int]:
        """
        Distribute paths based on the given evaluation function with logarithmic scaling and softmax-like distribution.
        """
        # Calculate the node pairs
        node_pairs = list(combinations(self.graph.nodes, 2))
        
        # Calculate the evaluation function and log values
        eval_values = {}
        for u, v in node_pairs:
            sp = nx.shortest_path(self.graph, source=u, target=v)
            sp_hop = len(sp) - 1
            sp_distance = nx.path_weight(self.graph, sp, weight='weight')
            f_uv = sp_hop + (sp_distance / self.max_length)
            eval_values[(u, v)] = f_uv

        # Normalize evaluation values to range [1, 2]
        min_eval = min(eval_values.values())
        max_eval = max(eval_values.values())
        normalized_values = {
            pair: 1 + (val - min_eval) / (max_eval - min_eval) for pair, val in eval_values.items()
        }
        # Apply softmax-like distribution
        total_weight = sum(normalized_values.values())
        weights = {
            pair: val / total_weight for pair, val in normalized_values.items()
        }

        # Calculate total paths to distribute
        total_paths = self.n_paths * len(node_pairs)

        # Allocate paths proportionally
        allocations = {
            pair: int(total_paths * weight) for pair, weight in weights.items()
        }

        # Adjust remaining paths
        allocated_paths = sum(allocations.values())
        remaining_paths = total_paths - allocated_paths

        # Sort by residuals and distribute remaining paths
        residuals = {
            pair: total_paths * weights[pair] - allocations[pair]
            for pair in node_pairs
        }
        sorted_pairs = sorted(residuals, key=residuals.get, reverse=True)

        for pair in sorted_pairs:
            if remaining_paths <= 0:
                break
            allocations[pair] += 1
            remaining_paths -= 1

        return allocations


class BasePathSelectorSinglePair(BasePathSelector):
    @abstractmethod
    def select_paths_single_pair(self, source: Any, target: Any) -> list[tuple[Any]]:
        """method to select paths for a single pair"""
        raise NotImplementedError

    def _select_all_paths(self) -> dict[tuple[Any, Any], list[tuple[Any]]]:
        """method to select paths for all pairs"""
        all_paths = {}
        nodes_pair = list(combinations(self.graph.nodes, 2))
        for source, target in nodes_pair:
            one_pair_paths = self.select_paths_single_pair(source, target)
            one_pair_paths_reverse = [list(reversed(path)) for path in one_pair_paths]
            all_paths[(source, target)] = one_pair_paths
            all_paths[(target, source)] = one_pair_paths_reverse
        return all_paths
