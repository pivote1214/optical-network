import os
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import RESULT_DIR


def make_metrics_table() -> str:
    file_names = os.listdir(RESULT_DIR / 'paths' / 'pickle')
    file_names = [path for path in file_names if path.endswith('.pickle')]
    file_names = sorted(file_names)
    metrics_table = pd.DataFrame(index=file_names)
    metrics_table.index.name = 'file_name'
    metrics_table = metrics_table.sort_index()

    for file_name in file_names:
        with open(RESULT_DIR / 'paths' / 'pickle' / file_name, 'rb') as f:
            content = pickle.load(f)
            for key, value in content['basic_info'].items():
                if type(value) == dict:
                    continue
                metrics_table.loc[file_name, key] = value
            for key, value in content['metrics'].items():
                if type(value) == dict:
                    continue
                metrics_table.loc[file_name, key] = value

    full_table_path = RESULT_DIR / 'paths' / 'metrics_table.csv'
    metrics_table.to_csv(full_table_path)

    return full_table_path


def plot_algorithm_comparison(
    metrics_df: pd.DataFrame, 
    k_value: int, 
    output_path: str
    ) -> None:
    """
    kDP, kSP, kBPの各アルゴリズムの評価指標を比較するグラフを作成する関数
    """
    df_filtered = metrics_df[metrics_df['k'] == k_value]

    # algorithm列がkBPの場合は，f'kBP_{alpha}'の形式に変更する
    df_filtered.loc[df_filtered['algorithm'] == 'kBP', 'algorithm'] = \
        df_filtered.loc[df_filtered['algorithm'] == 'kBP', 'algorithm'] + \
            '_' + df_filtered.loc[df_filtered['algorithm'] == 'kBP', 'alpha'].astype(str)

    # df_filteredの順をkDP, kBP_0.1,..., kBP_0.9, kSPにする
    sort_permutation = ['kSP'] + \
        [f'kBP_{round(alpha, 2)}' for alpha in np.arange(0.1, 1.0, 0.1)] + \
            ['kDP']
    j = 0
    for _, algorithm in enumerate(sort_permutation):
        if algorithm in df_filtered['algorithm'].values:
            df_filtered.loc[df_filtered['algorithm'] == algorithm, 'sort_key'] = j
            j += 1
    df_filtered = df_filtered.sort_values('sort_key')

    metrics_to_plot = ['elapsed_time', 'length_ave', 'hop_ave', 'similarity_ave', 
                       'path_num_ave', 'edge_usage_ave', 'edge_usage_std']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='algorithm', y=metric, data=df_filtered)
        plt.title(f'k={k_value}_{metric}')
        plt.ylabel(metric)
        plt.xlabel('Algorithm')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_path}_{metric}.png')
        plt.close()

    return None


if __name__ == '__main__':
    full_table_path = make_metrics_table()
    metrics_table = pd.read_csv(full_table_path, index_col='file_name')
    metrics_table = metrics_table.sort_index()

    for k in range(1, 6):
        output_path = RESULT_DIR / 'paths' / 'figures' / f'algorithm_comparison_k={k}'
        plot_algorithm_comparison(metrics_table, k, output_path)
