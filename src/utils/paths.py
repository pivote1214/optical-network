from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
PATHS_DIR = DATA_DIR / 'input' / 'paths'
RESULT_DIR = ROOT_DIR / 'results'
SRC_DIR = ROOT_DIR / 'src'
OPT_MODEL_DIR = SRC_DIR / 'optimize' / 'models'
