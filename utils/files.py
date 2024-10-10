import os
import sys

sys.path.append(os.path.abspath('../'))

import pickle
from datetime import datetime
from typing import Any

from utils.namespaces import PATHS_DIR, OUT_DIR


def save_pickle(object: Any, file_path: str) -> None:
    """
    objectをpickle形式でfile_pathに保存する関数

    Args:
        object (any): 保存するオブジェクト
        file_path (str): 保存先のファイルパス

    Returns:
        None
    """
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


def set_paths_file_path(
    algorithm: str, 
    network_name: str, 
    params: Any, 
    n_paths: int
    ) -> str:
    """
    グラフ上のすべてのパスを保存するファイルパスを設定する関数

    Args:
        algorithm (str): パス選択のアルゴリズム
        network_name (str): ネットワーク名
        params (dict[str, any]): アルゴリズムのパラメータ
        n_paths (int): パス数
    """
    # # get timestamp
    # timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S%f')
    # get params with '.' replaced by '_'
    params_name = [f"{key}={str(getattr(params, key)).replace('.', 'd')}" 
                   for key in params.__annotations__ 
                   if key not in ['timelimit']]
    paths_file_path = os.path.join(
        PATHS_DIR, algorithm, network_name, 
        *params_name, f"n-paths={n_paths}.pkl"
        )

    return paths_file_path


def set_result_dir(
    experiment_name: str, 
    algorithm: str, 
    network_name: str, 
    params: Any, 
    n_paths: int
    ) -> str:
    """
    グラフ上のすべてのパスを保存するファイルパスを設定する関数

    Args:
        algorithm (str): パス選択のアルゴリズム
        network_name (str): ネットワーク名
        params (dict[str, any]): アルゴリズムのパラメータ
        n_paths (int): パス数
    """
    # # get timestamp
    # timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    # get params with '.' replaced by '_'
    params_name = [f"{key}={str(getattr(params, key)).replace('.', 'd')}" 
                   for key in params.__annotations__ 
                   if key not in ['timelimit']]
    result = os.path.join(
        OUT_DIR, experiment_name, algorithm, network_name, 
        *params_name, f"n-paths={n_paths}"
        )

    return result
