import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PATHS_DIR = os.path.join(DATA_DIR, 'paths')
NETWORK_DIR = os.path.join(DATA_DIR, 'network')
OUT_DIR = os.path.join(ROOT_DIR, 'out')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
OPT_MODEL_DIR = os.path.join(SRC_DIR, 'optimize', 'models')
