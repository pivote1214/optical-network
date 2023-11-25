import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import PATHS_DIR, RESULT_DIR


def make_metrics_table() -> str:
    file_names = os.listdir(PATHS_DIR / 'NSF')
    file_names = sorted(file_names)
    metrics_table = pd.DataFrame(index=file_names)
    metrics_table.index.name = 'file_name'
    metrics_table = metrics_table.sort_index()

    for file_name in file_names:
        with open(PATHS_DIR/ 'NSF' / file_name, 'rb') as f:
            paths_data = pickle.load(f)
            for key, value in paths_data['parameters'].items():
                if type(value) == dict:
                    continue
                metrics_table.loc[file_name, key] = value
            for key, value in paths_data['metrics'].items():
                if type(value) == dict:
                    continue
                metrics_table.loc[file_name, key] = value

    full_table_path = RESULT_DIR / 'paths' / 'NSF' / 'metrics_table.csv'
    metrics_table.to_csv(full_table_path)

    return full_table_path


def plot_algorithm_comparison(
    metrics_df: pd.DataFrame, 
    k_value: int, 
    output_path: str
    ) -> None:
    """
    kDP, kSP, kSPwLOの各アルゴリズムの評価指標を比較するグラフを作成する関数
    """
    df_filtered = metrics_df[metrics_df['k'] == k_value]

    # algorithm列がkSPwLOの場合は，f'kSPwLO_{alpha}'の形式に変更する
    df_filtered.loc[df_filtered['algorithm'] == 'kSPwLO', 'algorithm'] = \
        df_filtered.loc[df_filtered['algorithm'] == 'kSPwLO', 'algorithm'] + \
            '_' + df_filtered.loc[df_filtered['algorithm'] == 'kSPwLO', 'alpha'].astype(str)

    # df_filteredの順をkDP, kSPwLO_0.1,..., kSPwLO_0.9, kSPにする
    sort_permutation = ['kSP'] + \
        [f'kSPwLO_{round(alpha, 2)}' for alpha in np.arange(0.1, 1.0, 0.1)] + \
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


if __name__ == '__main__':
    full_table_path = make_metrics_table()
    metrics_table = pd.read_csv(full_table_path, index_col='file_name')
    metrics_table = metrics_table.sort_index()

    for k in range(1, 6):
        output_path = RESULT_DIR / 'paths' / 'NSF' / 'figures' / f'algorithm_comparison_k={k}'
        plot_algorithm_comparison(metrics_table, k, output_path)
