import os
import pickle
from typing import Any

from src.namespaces import OUT_DIR, PATHS_DIR


def save_pickle(object: Any, file_path: str) -> None:
    """
    objectをpickle形式でfile_pathに保存する関数
    """
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


def load_pickle(file_path: str) -> Any:
    """
    pickle形式で保存されたobjectをfile_pathから読み込む関数
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def set_paths_dir(
    path_selector: str, 
    network_name: str, 
    n_paths: int, 
    **kwargs: Any,
    ) -> str:
    params_name = [f"{key}={str(value).replace('.', 'd')}" 
                   for key, value in kwargs.items()]
    params_name.sort()
    paths_dir = os.path.join(
        PATHS_DIR, 
        path_selector, 
        network_name, 
        *params_name, 
        f"n-paths={n_paths}"
        )
    return paths_dir


def set_result_dir(
    experiment_name: str, 
    algorithm: str, 
    network_name: str, 
    params: Any, 
    n_paths: int
    ) -> str:
    """
    グラフ上のすべてのパスを保存するディレクトリパスを設定する関数

    Args:
        algorithm (str): パス選択のアルゴリズム
        network_name (str): ネットワーク名
        params (dict[str, any]): アルゴリズムのパラメータ
        n_paths (int): パス数
    """
    params_name = [f"{key}={str(getattr(params, key)).replace('.', 'd')}" 
                   for key in params.__annotations__ 
                   if key not in ['timelimit']]
    result = os.path.join(
        OUT_DIR, experiment_name, algorithm, network_name, 
        *params_name, f"n-paths={n_paths}"
        )

    return result
