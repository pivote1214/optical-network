import os


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
TEST_DIR = os.path.join(ROOT_DIR, "test")
DATA_DIR = os.path.join(ROOT_DIR, "data")
NETWORK_DIR = os.path.join(DATA_DIR, "network")
GOOGLE_DRIVE_DIR = "/Users/pivote1214/Library/CloudStorage/GoogleDrive-yuta.m.12.1214@gmail.com/マイドライブ/optical-network"
OUT_DIR = os.path.join(GOOGLE_DRIVE_DIR, "out")
PATHS_DIR = os.path.join(DATA_DIR, "paths")
SRC_DIR = os.path.join(ROOT_DIR, "src")
