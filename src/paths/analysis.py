import os
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import RESULT_DIR

file_paths = os.listdir(RESULT_DIR / 'paths')
file_paths = [path for path in file_paths if path.endswith('.pickle')]
file_paths = sorted(file_paths)
metrics_table = pd.DataFrame(index=file_paths)
metrics_table.index.name = 'file_path'
metrics_table = metrics_table.sort_index()

for file_path in file_paths[-3:]:
    with open(RESULT_DIR / 'paths' / file_path, 'rb') as f:
        content = pickle.load(f)
        for key, value in content['basic_info'].items():
            if type(value) == dict:
                continue
            metrics_table.loc[file_path, key] = value
        for key, value in content['metrics'].items():
            if type(value) == dict:
                continue
            metrics_table.loc[file_path, key] = value

pprint.pprint(content)
        
# metrics_table.to_csv(RESULT_DIR / 'paths' / 'metrics_table.csv')
