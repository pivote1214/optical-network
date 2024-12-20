import os

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "NETWORK_DIR",
    "PATHS_DIR",
    "SRC_DIR",
]

ROOT_DIR    = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
PATHS_DIR   = os.path.join(DATA_DIR, "paths")
NETWORK_DIR = os.path.join(DATA_DIR, "network")
SRC_DIR     = os.path.join(ROOT_DIR, "src")
