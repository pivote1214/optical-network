import os
import pickle
from typing import Any

from src.namespaces import PATHS_DIR


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
        network_name, 
        path_selector, 
        *params_name, 
        f"n-paths={n_paths}"
        )
    return paths_dir
