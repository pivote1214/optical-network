import os

from src.paths.algorithms.base_algorithm import PathSelectionAlgorithm
from utils.network import calc_path_weight
from utils.namespaces import PATHS_DIR
from utils.files import save_pickle


class KShortestPaths(PathSelectionAlgorithm):
    def __init__(
        self, 
        graph_name: str, 
        n_paths: int, 
        params: dict = {'path_weight': 'physical-length'}, 
        length_limit: int = 6300
        ):
        super().__init__(graph_name, n_paths, params, length_limit)

    def select_k_paths_single_pair(
        self, 
        source: int, 
        target: int
        ) -> tuple[list[tuple[int]], float]:
        # source から target までの全てのシンプルパスを取得
        all_simple_paths = self._calc_all_simple_paths(source, target)
        # パスの重み順にソート
        if self.params['path_weight'] == 'physical-length':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path), 
                                                    calc_path_weight(self.graph, path, 'hop')))
        elif self.params['path_weight'] == 'hop':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path, 'hop'), 
                                                    calc_path_weight(self.graph, path)))
        elif self.params['path_weight'] == 'expected-used-slots':
            all_simple_paths.sort(key=lambda path: (calc_path_weight(self.graph, path, 'expected-used-slots'), 
                                                    calc_path_weight(self.graph, path, 'hop'), 
                                                    calc_path_weight(self.graph, path)))
        else:
            raise ValueError('path_weight must be "physical-length", "hop" or "expected-used-slots')
        # 重みの小さなk本のパスを取得
        k_paths = all_simple_paths[:self.n_paths]

        return k_paths

    def save_selected_paths_all_pairs(self) -> None:
        all_paths = self.select_k_paths_all_pairs()
        output_file = os.path.join(
            PATHS_DIR, 
            'k-shortest-paths', 
            self.graph_name, 
            f'path_weight_{self.params["path_weight"]}', 
            f'n-paths_{self.n_paths}.pkl'
            )
        save_pickle(all_paths, output_file)
