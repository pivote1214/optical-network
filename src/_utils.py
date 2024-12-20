from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from src.graph import calc_path_weight
from src.rsa.demands import Demand
from src.rsa.optical_network import Width

__all__ = ["calc_required_slots"]


def calc_required_slots(
    demand: Demand,
    path: list[Any],
    graph: nx.DiGraph,
    modulation_formats: list[tuple[int, int]],
    width: Width,
    traffic_vol_bpsk: int,
    ) -> int:
    path_length = calc_path_weight(graph, path, metrics="physical-length")
    modulation_level = get_modulation_level(path_length, modulation_formats)
    required_slots = np.ceil(
        (
            np.ceil((demand.traffic_vol / (modulation_level * traffic_vol_bpsk)))
            * width.optical_carrier
            + 2 * width.guard_band
        )
        / width.frequency_slot
    )
    required_slots = int(required_slots)

    return required_slots


def get_modulation_level(
    path_length: int, modulation_formats: list[tuple[int, int]]
    ) -> int:
    """
    パスの長さから変調方式を選択する

    Parameters
    ----------
    path_length : int
        パスの長さ
    modulation_format : dict, optional
        変調方式の最大伝送距離と変調レベル

    Returns
    -------
    int
        選択された変調方式のレベル

    Raises
    ------
    ValueError
        パスの長さが長すぎる場合
    """
    for length_limit, modulation_level in modulation_formats:
        if path_length <= length_limit:
            return modulation_level
    raise ValueError("Path length is too long.")


def plot_dendrogram(Z: np.ndarray) -> None:
    dendrogram(Z)
    plt.xlabel("Path Index")
    plt.ylabel("Distance")
    plt.show()


def output_latex_table(
    df: pd.DataFrame,
    caption: Optional[str] = "Table Caption",
    label: Optional[str] = "tab:label",
    file_path: Optional[str] = None
    ) -> None:
    """
    Outputs a DataFrame in a LaTeX table format with customizable multicolumns.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert to LaTeX table format.
        caption (str): Caption for the table.
        label (str): LaTeX label for referencing the table.
        file_path (str): Path to save the LaTeX table output. If None, prints to console.
    """
    num_levels = df.columns.nlevels
    n_col = len(df.columns)

    # Initialize LaTeX code for the table
    latex_code = "\\begin{table}[htb]\n"
    latex_code += "  \\centering\n"
    latex_code += f"  \\caption{{{caption}}}\n"
    latex_code += f"  \\label{{{label}}}\n"
    latex_code += "  \\begin{tblr}{\n"
    latex_code += f"    colspec={{c|{'r' * n_col}}}, \n"
    latex_code += "    hline{1,Z} = {1.2pt},\n"
    latex_code += "    }\n"

    # Generate header rows
    header_rows = []

    for level in range(num_levels):
        headers = df.columns.get_level_values(level)

        row = "    & "
        i = 0
        while i < len(headers):
            colspan = 1
            while i + colspan < len(headers) and headers[i] == headers[i + colspan]:
                colspan += 1

            if level == num_levels - 1:
                # Last level: output column names without grouping
                row += f"\\textbf{{{headers[i]}}}"
            else:
                # Group headers and use \SetCell
                if colspan > 1:
                    row += f"\\SetCell[c={colspan}]{{c}} \\textbf{{{headers[i]}}}"
                else:
                    row += f"\\textbf{{{headers[i]}}}"

            i += colspan
            if i < len(headers):
                row += " & " * colspan
        row += " \\\\\n"
        header_rows.append(row)

    # Add header rows to LaTeX code
    for header_row in header_rows:
        latex_code += header_row

    latex_code += "    \\hline\n"

    # Add data rows
    for index, row_data in df.iterrows():
        row_str = " & ".join(f"{value}" for value in row_data)
        latex_code += f"    {index} & {row_str} \\\\\n"

    # Close LaTeX table environment
    latex_code += "  \\end{tblr}\n"
    latex_code += "\\end{table}\n"

    # Output or save LaTeX code
    if file_path:
        with open(file_path, "w") as file:
            file.write(latex_code)
    else:
        print(latex_code)
