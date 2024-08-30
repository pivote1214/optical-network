import os
import sys
sys.path.append(os.path.abspath('../'))

import math
import logging
from pprint import pprint

import gurobipy as gp
import networkx as nx

from utils.namespaces import TEST_DIR
from utils.network import load_network
from src.demands.demands import gen_all_demands_offline


# 定数の定義
BR_M_OC = 50
MAX_FS = 320
OC_BANDWIDTH = 37.5
GUARD_BAND = 12.5
FS_BANDWIDTH = 12.5
REACH_R = 6300
THETA = 1e-5
S_I_SET = [1]
TIME_LIMITS = {
    'SLC_RMS': 30,
    'SLC_SA_S': 3600,
    'SLC_S': 7200,
    'SLC_C': 7200,
}


def setup_logging():
    """ログ設定の初期化"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=os.path.join(TEST_DIR, "log", "wang.log")),
        ]
    )
    return logging.getLogger(__name__)

def calculate_path_length(path, graph):
    """指定されたパスの長さを計算"""
    return sum(graph[n1][n2]['weight'] for n1, n2 in zip(path[:-1], path[1:]))

def select_modulation_format(path_length):
    """パスの長さに応じた変調方式を選択"""
    if path_length <= 600:
        return 4  # DP-16QAM
    elif path_length <= 1200:
        return 3  # DP-8QAM
    elif path_length <= 3500:
        return 2  # DP-QPSK
    elif path_length <= 6300:
        return 1  # DP-BPSK
    return 0  # 到達不可距離

def required_fs(traffic_size, modulation):
    """必要な周波数スロット数を計算"""
    OCs = math.ceil(traffic_size / (BR_M_OC * modulation))
    OCs_per_core = math.ceil(OCs / S_I_SET[0])
    num_FS = math.ceil((OCs_per_core * OC_BANDWIDTH + GUARD_BAND) / FS_BANDWIDTH)
    return num_FS, OCs_per_core

def common_links(path1, path2):
    """2つのパスに共通するリンクを見つける"""
    common_edges = [(n1, n2) for n1, n2 in zip(path1[:-1], path1[1:]) if (n1 in path2 and path2[path2.index(n1) + 1] == n2)]
    return common_edges

def read_requests(seed, network_name, graph, n_demands):
    R = gen_all_demands_offline(
        graph=graph, N_DEMANDS=n_demands, 
        demands_population=[50, 100, 150, 200], seed=seed
    )
    return R

def calculate_ksp_and_modulation(graph, n_paths):
    nodes = list(graph.nodes())
    physical_ksp_sd_set = {}
    hysical_m = {}
    for s in nodes:
        for d in nodes:
            if s != d:
                KSP_sd = []
                all_physical_path_sd = list(nx.shortest_simple_paths(graph, s, d))
                sorted_all_physical_path_sd = sorted(
                    all_physical_path_sd , 
                    key = lambda physical_p : (len(physical_p), calculate_path_length(physical_p, graph))
                    )
                for physical_path in sorted_all_physical_path_sd[:n_paths]:
                    '''Adaptive modulation format for a lightpath p'''
                    m_physical_path = select_modulation_format(calculate_path_length(physical_path, graph))
                    if m_physical_path != 0:
                        KSP_sd.append(tuple(physical_path))
                        hysical_m[tuple(physical_path)] = m_physical_path
                physical_ksp_sd_set[s , d] = KSP_sd
    return physical_ksp_sd_set, hysical_m


def initialize_simulation(R, physical_ksp_sd_set, physical_m):
    """シミュレーションの初期化"""
    AP, M, P, F_rp, O_rp = {}, {}, {}, {}, {}
    p = 1
    for r, (s, d, t) in R.items():
        P_r = []
        for path in physical_ksp_sd_set[s, d]:
            AP[p] = path
            M[p] = physical_m[path]
            P_r.append(p)
            p += 1
        P[r] = P_r
    
    for r, paths in P.items():
        for p_r in paths:
            F_rp[r, p_r], O_rp[r, p_r] = required_fs(R[r][2], M[p_r])
    return AP, M, P, F_rp, O_rp

def run_ilp_slc_relaxation(logger, time_limit, R, P, AP, F_rp, E, SG):
    """ILP_SLC_Relaxモデルの実行"""
    logger.info("ILP_SLC_Relaxモデル開始")

    model = gp.Model('SLC_RMS')
    model.Params.LogToConsole = 0

    FS_use_ILP_SLC_RMS = model.addVar(vtype=gp.GRB.INTEGER)
    x_slc_rms = {(r, p): model.addVar(vtype=gp.GRB.BINARY) for r in R for p in P[r]}
    y_slc_rms = {(p, e, sg): model.addVar(vtype=gp.GRB.BINARY) for p in AP for e in E for sg in SG}

    model.setObjective(FS_use_ILP_SLC_RMS, gp.GRB.MINIMIZE)
    
    # 制約の追加
    for r in R:
        model.addConstr(gp.quicksum(x_slc_rms[r, p] for p in P[r]) == 1)
    
    for r in R:
        for p in P[r]:
            for e in common_links(AP[p], AP[p]):
                model.addConstr(gp.quicksum(y_slc_rms[p, e, sg] for sg in SG) == x_slc_rms[r, p])
    
    for e in E:
        for sg in SG:
            model.addConstr(FS_use_ILP_SLC_RMS >= gp.quicksum(F_rp[r, p] * y_slc_rms[p, e, sg] for r in R for p in P[r] if e in common_links(AP[p], AP[p])) - 1)
    
    model.setParam('TimeLimit', time_limit)
    model.optimize()

    # print("Demands")
    # pprint(R)

    undone = model.MIPGap > 0
    solution = [model.ObjVal, model.ObjBound, model.MIPGap, model.Runtime]
    best_PEG = {r: {p: {e: sg for e, sg in zip(common_links(AP[p], AP[p]), SG) if int(y_slc_rms[p, e, sg].x + THETA) == 1}} for r in R for p in P[r]}

    # return best_PEG, max(model.ObjVal, model.ObjBound), solution, undone
    return best_PEG, model.ObjVal, solution, undone

def main(network_name: str, n_paths: int, n_demands: int):
    logger = setup_logging()
    # ネットワークとパラメータの初期化
    graph = load_network(network_name)

    E = list(graph.edges())
    S = [1]
    SG = [i for i in range(1, 1 + int(len(S) / S_I_SET[0]))]

    # KSPとモジュレーションの計算
    physical_ksp_sd_set, physical_m = calculate_ksp_and_modulation(graph, n_paths)
    
    lower_bound_obj_vals = []
    # リクエストの読み込みとシミュレーションの初期化
    for seed in range(2, 21, 2):  # 10回のイテレーション
        # print('-' * 80)
        # print(f"Seed: {seed}")
        R = read_requests(seed, network_name, graph, n_demands)
        AP, M, P, F_rp, O_rp = initialize_simulation(R, physical_ksp_sd_set, physical_m)
        # print("Demands")
        # pprint(R)
        # print("Paths")
        # pprint(AP)
        # print("Number of Slots")
        # pprint(M)
        # ILP_SLC_Relaxモデルの実行
        best_PEG, lbound, solution, undone = run_ilp_slc_relaxation(
            logger, TIME_LIMITS['SLC_RMS'], R, P, AP, F_rp, E, SG
        )
        lower_bound_obj_vals.append(int(lbound) + 1)
    
    # 結果の出力
    print("Lower Bound Objective Values")
    pprint(lower_bound_obj_vals)

if __name__ == "__main__":
    main("JPN12")
