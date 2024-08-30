import os
import sys
sys.path.append(os.path.abspath('../'))

import math
import random
import pickle
import logging
from pprint import pformat, pprint

import networkx as nx
import numpy as np
import gurobipy as gp

from utils.namespaces import TEST_DIR


def Length_path(physical_p):
    p_length = 0
    Num_n_p = len(physical_p)
    for n_index in range(Num_n_p - 1):
        n1 = physical_p[n_index]
        n2 = physical_p[n_index + 1]
        e_length = G[n1][n2]['weight']
        p_length += e_length
    return p_length

def Adaptive_modulation(path_length):    
    if path_length <= 600: #DP-16QAM
        return 4
    elif path_length <= 1200: #DP-8QAM
        return 3
    elif path_length <= 3500: #DP-QPSK
        return 2
    elif path_length <= 6300: #DP-BPSK
        return 1
    else:
        return 0 #Out of reachable distance


def FS_required(traffic_size, modulation):
    OCs = math.ceil(traffic_size/(BR_M_OC*modulation))
    OCs_per_core = math.ceil(OCs/s_i)
    num_FS = math.ceil((OCs_per_core*OC_Bandwidth + Guardband)/FS_Bandwidth)
    return num_FS, OCs_per_core

def Check_common_links(physical_p1, physical_p2):
    Num_n_p1 = len(physical_p1)
    Num_n_p2 = len(physical_p2)
    for n_index_p1 in range(Num_n_p1 - 1):
        n = physical_p1[n_index_p1]
        if n in physical_p2:
            n_index_p2 = physical_p2.index(n)
            if n_index_p2 != Num_n_p2 - 1:
                n_next_p1 = physical_p1[n_index_p1 + 1]
                n_next_p2 = physical_p2[n_index_p2 + 1]
                if n_next_p1 == n_next_p2:
                    return 1
    return 0


def Common_links(physical_p1, physical_p2):
    Common_E = []
    Num_n_p1 = len(physical_p1)
    Num_n_p2 = len(physical_p2)
    for n_index_p1 in range(Num_n_p1 - 1):
        n = physical_p1[n_index_p1]
        if n in physical_p2:
            n_index_p2 = physical_p2.index(n)
            if n_index_p2 != Num_n_p2 - 1:
                n_next_p1 = physical_p1[n_index_p1 + 1]
                n_next_p2 = physical_p2[n_index_p2 + 1]
                if n_next_p1 == n_next_p2:
                    e = (n, n_next_p1)
                    Common_E.append(e)
    return Common_E

def Links_in_path(physical_p):
    Num_n_p = len(physical_p)
    E_in_p = []
    for n_index in range(Num_n_p - 1):
        n1 = physical_p[n_index]
        n2 = physical_p[n_index + 1]
        e = (n1, n2)
        E_in_p.append(e)
    return E_in_p

def Traffic_generation_one_r(Generation_principle, value1, value2):
    if Generation_principle == 'Random':
        return random.randint(value1, value2)
    if Generation_principle == 'Gaussian':
        return int(np.random.normal(value1, value2))
    
def Traffic_generation_R(Generation_principle, R_num, r_value1, r_value2):
    R = {}
    r_index = 1
    while r_index <= R_num: 
        s = V[random.randint(0, len(V) - 1)]
        d = V[random.randint(0, len(V) - 1)]
        if s != d:
            R[r_index] = [s, d, Traffic_generation_one_r(Generation_principle, r_value1, r_value2)]
            r_index += 1
    return R

def Traffic_generation_R_perpair(Generation_principle, r_num_per_pair, r_value1, r_value2):
    R = {}
    r_index = 1
    for s in V:
        for d in V:
            if s != d:
                for num_r in range(r_num_per_pair):
                    R[r_index] = [s, d, Traffic_generation_one_r(Generation_principle, r_value1, r_value2)]
                    r_index += 1
    return R


def KSP_ASD_and_mudulation_AP(Network):
    Physical_KSP_sd_set = {}
    Physical_M = {}
    for s in V:
        for d in V:
            if s != d:
                KSP_sd = []
                All_physical_path_sd = list(nx.shortest_simple_paths(Network , s , d))
                Sorted_All_physical_path_sd = sorted(All_physical_path_sd , key = lambda physical_p : (len(physical_p), Length_path(physical_p)))
                for physical_path in Sorted_All_physical_path_sd[:2]:
                    '''Adaptive modulation format for a lightpath p'''
                    m_physical_path = Adaptive_modulation(Length_path(physical_path))
#                    '''Constant modulation format for all lightpaths'''
#                    m_physical_path = Constant_modulation()
                    if m_physical_path != 0:
                        KSP_sd.append(tuple(physical_path))
                        Physical_M[tuple(physical_path)] = m_physical_path
                Physical_KSP_sd_set[s , d] = KSP_sd
    return Physical_KSP_sd_set, Physical_M

def Simulation_initialization():
    AP = {}
    M = {}
    P = {}
    F_rp = {}
    O_rp = {}
    p = 1
    for r in R:
        P_r = []
        [s, d, t] = R[r]
        Physical_KSP_r = Physical_KSP_sd_set[s, d]
        for Physical_p in Physical_KSP_r:
            AP[p] = Physical_p
            M[p] = Physical_M[Physical_p]
            P_r.append(p)
            p += 1
        P[r] = P_r
    for r in R:
        for p_r in P[r]:
            F_rp[r, p_r], O_rp[r , p_r] = FS_required(R[r][2], M[p_r])
    return AP, M, P, F_rp, O_rp

'''ILP formulations'''

'''ILP-SLC-decomposed models'''

def ILP_SLC_Relax(Time_Limit_SLC_RMS):
    
    '''ILP_RMS'''
    
    print("ILP_SLC_Relax model開始")
    
    model_SLC_RMS = gp.Model('SLC_RMS')
    
    '''Variables'''
    
    FS_use_ILP_SLC_RMS = model_SLC_RMS.addVar(vtype=gp.GRB.INTEGER)
    
    x_slc_rms = {}
    for r in R:
        for p in P[r]:
            x_slc_rms[p] = model_SLC_RMS.addVar(vtype=gp.GRB.BINARY)
    
    y_slc_rms = {}
    for r in R:
        for p in P[r]:
            for e in E:
                for sg in SG:
                    y_slc_rms[p, e, sg] = model_SLC_RMS.addVar(vtype=gp.GRB.BINARY)
                
    '''Objective Function'''
        
    model_SLC_RMS.setObjective(FS_use_ILP_SLC_RMS, gp.GRB.MINIMIZE)
    
    '''Constraints'''
    
    for r in R:
        x_sum = gp.quicksum(x_slc_rms[p] for p in P[r])
        model_SLC_RMS.addConstr(x_sum == 1)
    
    for r in R:
        for p in P[r]:
            for e in Links_in_path(AP[p]):
                y_sum = gp.quicksum(y_slc_rms[p, e, sg] for sg in SG)
                model_SLC_RMS.addConstr(y_sum == x_slc_rms[p])
                
    for e in E:
        for sg in SG:
            y_sum = gp.quicksum(F_rp[r , p]*y_slc_rms[p, e, sg] for r in R for p in P[r] if e in Links_in_path(AP[p]))
            model_SLC_RMS.addConstr(FS_use_ILP_SLC_RMS - y_sum + 1 >= 0)
            
    model_SLC_RMS.setParam('TimeLimit', Time_Limit_SLC_RMS)
    model_SLC_RMS.optimize()
    
    if model_SLC_RMS.MIPGap > 0:
        undone_SLC_RMS = 1
    else:
        undone_SLC_RMS = 0
    
    Solution_ILP_SLC_RMS = [model_SLC_RMS.ObjVal, model_SLC_RMS.Objbound, model_SLC_RMS.MIPGap, model_SLC_RMS.RunTime]
    
    Best_PEG = {}
    for r in R:
        ifuse = {}
        for p in P[r]:
            ifuse[p] = 0
        Best_PEG[r] = {}
        for p in P[r]:
            for e in Links_in_path(AP[p]):
                for sg in SG:
                    if int(y_slc_rms[p, e, sg].x + theta) == 1:
                        ifuse[p] = 1
        for p in P[r]:
            if ifuse[p] == 1:
                Best_PEG[r][p] = {}
                for e in Links_in_path(AP[p]):
                    Best_PEG[r][p][e] = {}
                    for sg in SG:
                        if int(y_slc_rms[p, e, sg].x + theta) == 1:
                            Best_PEG[r][p][e] = sg
    
    if model_SLC_RMS.ObjBound < model_SLC_RMS.ObjVal:
        LBound_ILP_SLC_RMS = model_SLC_RMS.ObjBound
    else:
        LBound_ILP_SLC_RMS = model_SLC_RMS.ObjVal
    
    return Best_PEG, LBound_ILP_SLC_RMS, Solution_ILP_SLC_RMS, undone_SLC_RMS

    # return model_SLC_RMS.ObjVal

def ILP_SLC_decomposed_Slot(Time_Limit_SLC_SA_S):
                   
    '''ILP_SA_S'''
    
    print("ILP_SLC_decomposed_Slot model開始")
    
    model_SLC_SA_S = gp.Model('SLC_SA_S')
    
    '''Variables'''
    
    FS_max_ILP_SLC_SA_S = model_SLC_SA_S.addVar(vtype=gp.GRB.INTEGER)
    
    f_slc_sa_s = {}
    for r in R:
        f_slc_sa_s[r] = model_SLC_SA_S.addVar(vtype=gp.GRB.INTEGER)        
    
    xi_slc_sa_s = {}
    for r1 in R:
        for r2 in R:
            if r1 != r2:
                xi_slc_sa_s[r1, r2] = model_SLC_SA_S.addVar(vtype=gp.GRB.BINARY)
    model_SLC_SA_S.update()
    
    B_p = {}
    B_e = {}
    B_sg = {}
    for r in R:
        B_p[r] = list(Best_PEG[r].keys())[0]
        B_e[r] = []
        B_sg[r] = []
        for e_num in range(len(Best_PEG[r][B_p[r]])):
            B_e[r].append(list(Best_PEG[r][B_p[r]].keys())[e_num])
            B_sg[r].append(Best_PEG[r][B_p[r]][B_e[r][e_num]])
    '''Objective Function'''
        
    model_SLC_SA_S.setObjective(FS_max_ILP_SLC_SA_S, gp.GRB.MINIMIZE) 
    
    '''Constraints'''
    
    for r1 in R:
        for r2 in R:
            if r1 != r2:
                model_SLC_SA_S.addConstr(xi_slc_sa_s[r1, r2] + xi_slc_sa_s[r2, r1] == 1)
            
    for r in R:
        p = B_p[r]
        model_SLC_SA_S.addConstr(FS_max_ILP_SLC_SA_S - f_slc_sa_s[r] - F_rp[r, p] + 1 >= 0)
    
    for r1 in R:
        for r2 in R:
            if r1 != r2:
                p1 = B_p[r1]
                p2 = B_p[r2]
                for e in Common_links(AP[p1], AP[p2]):
                    sg1 = B_sg[r1][B_e[r1].index(e)]
                    sg2 = B_sg[r2][B_e[r2].index(e)]
                    if sg1 == sg2:
                        model_SLC_SA_S.addConstr(f_slc_sa_s[r2] + F_rp[r2 , p2] - f_slc_sa_s[r1] - Max_FS*(1 - xi_slc_sa_s[r1, r2]) <= 0)

    model_SLC_SA_S.addConstr(FS_max_ILP_SLC_SA_S >= LBound_ILP_SLC_RMS)
    
    model_SLC_SA_S.setParam('TimeLimit', Time_Limit_SLC_SA_S)            
    model_SLC_SA_S.optimize()
    
    if model_SLC_SA_S.MIPGap > 0:
        undone_SLC_SA_S = 1
    else:
        undone_SLC_SA_S = 0
    
    Solution_ILP_SLC_S_decomposed = [model_SLC_SA_S.ObjVal, model_SLC_SA_S.Objbound, model_SLC_SA_S.MIPGap, model_SLC_SA_S.RunTime]
    Assignment_ILP_SLC_S_decomposed = {}
    for r in R:
        p = B_p[r]
        ILP_SLC_SG_r = {}
        for e in B_e[r]:
            sg = B_sg[r][B_e[r].index(e)]
            ILP_SLC_SG_r[e] = sg
        Assignment_ILP_SLC_S_decomposed[r] = p, ILP_SLC_SG_r, int(f_slc_sa_s[r].x + theta), int(f_slc_sa_s[r].x + theta) + F_rp[r, p] - 1
                    
    return Solution_ILP_SLC_S_decomposed, Assignment_ILP_SLC_S_decomposed, undone_SLC_SA_S

#def ILP_SLC_decomposed_Channel(Time_Limit_SLC_SA_C):
#                   
#    '''ILP_SA_C'''
#    
#    print("ILP_SLC_decomposed_Channel model開始")
#    
#    model_SLC_SA_C = gp.Model("SLC_SA_C")
#    
#    '''Variables'''
#    x_slc_sa_c = {}
#    for r in R:
#        for p in P[r]:
#            for c in C[r, p]:
#                x_slc_sa_c[p, c] = model_SLC_SA_C.addVar(vtype = gp.GRB.BINARY)
#    
#    y_slc_sa_c = {}
#    for r in R:
#        for p in P[r]:
#            for e in E:
#                for sg in SG:
#                    for c in C[r, p]:
#                        y_slc_sa_c[p, e, sg, c] = model_SLC_SA_C.addVar(vtype = gp.GRB.BINARY)
#    
#    FS_max_ILP_SLC_SA_C = model_SLC_SA_C.addVar(vtype = gp.GRB.INTEGER)
#    
#    model_SLC_SA_C.update()
#    
#    B_p = {}
#    B_e = {}
#    B_sg = {}
#    for r in R:
#        B_p[r] = list(Best_PEG[r].keys())[0]
#        B_e[r] = []
#        B_sg[r] = []
#        for e_num in range(len(Best_PEG[r][B_p[r]])):
#            B_e[r].append(list(Best_PEG[r][B_p[r]].keys())[e_num])
#            B_sg[r].append(Best_PEG[r][B_p[r]][B_e[r][e_num]])
#    
#    '''Objective Function'''
#    
#    model_SLC_SA_C.setObjective(FS_max_ILP_SLC_SA_C, gp.GRB.MINIMIZE)
#    
#    '''Constraints'''
#    for r in R:
#        p = B_p[r]
#        Path_channel_R = gp.quicksum(x_slc_sa_c[p, c] for c in C[r, p])
#        model_SLC_SA_C.addConstr(Path_channel_R == 1)
#    
#    for r in R:
#        p = B_p[r]
#        for e in B_e[r]:
#            sg = B_sg[r][B_e[r].index(e)]
#            for c in C[r, p]:
#                Link_channel_R = y_slc_sa_c[p, e, sg, c]
#                model_SLC_SA_C.addConstr(Link_channel_R == x_slc_sa_c[p, c])
#                
#    for e in E:
#        for sg in SG:
#            for f in range(int(LBound_ILP_SLC_RMS + 1) + 10):
#                No_overlap = gp.quicksum(gamma[r, p, c, f] * y_slc_sa_c[p, e, sg, c] for r in R for p in P[r] if e in Links_in_path(AP[p]) for c in C[r, p])
#                model_SLC_SA_C.addConstr(No_overlap - 1 <= 0)
#                
#    for r in R:
#        p = B_p[r]
#        for e in B_e[r]:
#            sg = B_sg[r][B_e[r].index(e)]
#            model_SLC_SA_C.addConstr(gp.quicksum(F_rp[r, p]*y_slc_sa_c[p, e, sg, c] + c*y_slc_sa_c[p, e, sg, c] for c in C[r, p]) - 1 - FS_max_ILP_SLC_SA_C <= 0)
#    
#    model_SLC_SA_C.addConstr(FS_max_ILP_SLC_SA_C >= LBound_ILP_SLC_RMS)
#        
#    model_SLC_SA_C.setParam('TimeLimit', Time_Limit_SLC_SA_C) 
#    model_SLC_SA_C.optimize()
#    
##    for e in E:
##        for sg in SG:
##            for r in R:
##                for p in P[r]:
##                    for c in C[r,p]:
##                        if y_slc_sa_c[p,e,sg,c].x == 1:
##                            print(r,p,e,sg,c,c+F_rp[r,p]-1)
#    
#    
#    if model_SLC_SA_C.MIPGap > 0:
#        undone_SLC_SA_C = 1
#    else:
#        undone_SLC_SA_C = 0
#    
#    Solution_ILP_SLC_C_decomposed = [model_SLC_SA_C.ObjVal, model_SLC_SA_C.Objbound, model_SLC_SA_C.MIPGap, model_SLC_SA_C.RunTime]
#    Assignment_ILP_SLC_C_decomposed = {}
#    for r in R:
#        p = B_p[r]
#        ILP_SLC_SG_r = {}
#        for e in B_e[r]:
#            sg = B_sg[r][B_e[r].index(e)]
#            ILP_SLC_SG_r[e] = sg
#        for e in Links_in_path(AP[p]):
#            for sg in SG:
#                for c in C[r, p]:
#                    if int(y_slc_sa_c[p, e, sg, c].x + theta) == 1:
#                        Assignment_ILP_SLC_C_decomposed[r] = p, ILP_SLC_SG_r, c, c + F_rp[r, p] - 1
#                    
#    return Solution_ILP_SLC_C_decomposed, Assignment_ILP_SLC_C_decomposed, undone_SLC_SA_C


'''ILP-SLC-C (path/channel model)'''

def Find_Channel_bound(Solution_ILP_SLC_S_decomposed):
    if Solution_ILP_SLC_S_decomposed[0] < Max_FS:
        Channel_bound = Solution_ILP_SLC_S_decomposed[0]
    else:
        Channel_bound = Max_FS
    return Channel_bound

def ILP_SLC_Channel(Time_Limit_SLC_C):
    print("ILP_SLC_Channel model開始")
    
    model_SLC_C = gp.Model("SLC_C")
    
    """Variables"""
    
    FS_max_ILP_SLC_C = model_SLC_C.addVar(vtype=gp.GRB.INTEGER)
       
    x_slc_c = {}
    for r in R:
        for p in P[r]:
            for c in C[r, p]:
                x_slc_c[p, c] = model_SLC_C.addVar(vtype = gp.GRB.BINARY)
            
    y_slc_c = {}
    for r in R:
        for p in P[r]:
            for e in E:
                for sg in SG:
                    for c in C[r, p]:
                        y_slc_c[p, e, sg, c] = model_SLC_C.addVar(vtype = gp.GRB.BINARY)
                        
    model_SLC_C.update()
        
    '''Set objective'''
   
    model_SLC_C.setObjective(FS_max_ILP_SLC_C, gp.GRB.MINIMIZE)
                
    '''Add constraint'''
    
    for r in R:
        Path_channel_R = gp.quicksum(x_slc_c[p, c] for p in P[r] for c in C[r, p])
        model_SLC_C.addConstr(Path_channel_R == 1)

    for r in R:
        for p in P[r]:
            for e in Links_in_path(AP[p]):
                for c in C[r, p]:
                    Link_channel_R = gp.quicksum(y_slc_c[p, e, sg, c] for sg in SG)
                    model_SLC_C.addConstr(Link_channel_R == x_slc_c[p, c])
                    
    for e in E:
        for sg in SG:
            for f in range(int(Channel_bound) + 1):
                No_overlap = gp.quicksum(gamma[r, p, c, f]*y_slc_c[p, e, sg, c] for r in R for p in P[r] if e in Links_in_path(AP[p]) for c in C[r,p])
                model_SLC_C.addConstr(No_overlap - 1 <= 0)
                
    for r in R:
        for e in E:
            for sg in SG:
                model_SLC_C.addConstr(gp.quicksum(F_rp[r,p]*y_slc_c[p, e, sg, c] + c*y_slc_c[p, e, sg, c] for p in P[r] for c in C[r,p]) - 1 - FS_max_ILP_SLC_C <= 0)
    
    model_SLC_C.addConstr(FS_max_ILP_SLC_C - LBound_ILP_SLC_RMS >= 0)

    '''input initial solution'''
    
    for r in R:
        p, Able_SG_r, FS_start_r, FS_end_r = Assignment_ILP_SLC_S_decomposed[r]
        c = FS_start_r
        x_slc_c[p, c].start = 1
        for e in Links_in_path(AP[p]):
            sg_e = Able_SG_r[e]
            y_slc_c[p, e, sg_e, c].start = 1
    
    model_SLC_C.setParam('TimeLimit', Time_Limit_SLC_C)                                
    model_SLC_C.optimize()
    
    if model_SLC_C.MIPGap > 0:
        undone_SLC_C = 1
    else:
        undone_SLC_C = 0
    
    Solution_ILP_SLC_C = [model_SLC_C.ObjVal, model_SLC_C.Objbound, model_SLC_C.MIPGap, model_SLC_C.RunTime]
    Assignment_ILP_SLC_C = {}
    for r in R:
        for p in P[r]:
            for c in C[r, p]:
                if int(x_slc_c[p, c].x + theta) == 1:
                    ILP_SLC_SG_C_r = {}
                    for e in Links_in_path(AP[p]):
                        for sg in SG:
                            if int(y_slc_c[p, e, sg, c].x + theta) == 1:
                                ILP_SLC_SG_C_r[e] = sg
                    
                    for e in Links_in_path(AP[p]):
                        for sg in SG:
                            for c in C[r, p]:
                                if int(y_slc_c[p, e, sg, c].x + theta) == 1:
                                    Assignment_ILP_SLC_C[r] = p, ILP_SLC_SG_C_r, c, c + F_rp[r, p] - 1
                            
    return Solution_ILP_SLC_C, Assignment_ILP_SLC_C, undone_SLC_C


'''Requests write and read'''

def Request_maker(T_volume_type, R_num, r_value1, r_value2, times):
    for iteration_num in range(times):
        R = Traffic_generation_R(T_volume_type, R_num, r_value1, r_value2)    
        xls = xlwt.Workbook()
        sht1 = xls.add_sheet('Sheet1')
        sht1.write(0,0,'R')
        sht1.write(0,1,'start')
        sht1.write(0,2,'destination')
        sht1.write(0,3,'Traffic')
        for j in range(1,R_num+1):
            sht1.write(j,0,j)
            sht1.write(j,1,R[j][0])
            sht1.write(j,2,R[j][1])
            sht1.write(j,3,R[j][2])
        xls.save('D:\\学術関係\SDM-SLC_models\Requests\\'+'request_data_'+str(R_num)+'_'+str(iteration_num)+'.xls')

def Request_reader(seed):
    file_path = os.path.join(TEST_DIR, f"test_seed={seed}.pkl")
    with open(file_path, "rb") as f:
        R = pickle.load(f)
    return R

def Channels_generation(F_rp, Channel_bound):
    C = {}
    for r in R:
        for p in P[r]:
            flag = 0
            C[r, p] = {}
            for c in range(int(Channel_bound) + 1 - F_rp[r, p] + 1):
                sign = flag
                C[r, p][c] = {}
                for f in range(F_rp[r, p]):
                    C[r, p][c][f] = sign
                    sign += 1
                flag += 1
    
    gamma = {}
    for r in R:
        for p in P[r]:
            for c in C[r, p]:
                for f in range(Max_FS):
                    gamma[r, p, c, f] = 0
                    for k in C[r, p][c]:
                        if f == C[r, p][c][k]:
                            gamma[r, p, c, f] = 1
    return C, gamma

'''Simulation initialization''' 
   
'''Network topologies'''
NSF = [(1, 2, 1100),(1, 3, 1600),(1, 8, 2800),(2, 3, 600),(2, 4, 1000),(3, 6, 2000),(4, 5, 600),(4, 11, 2400),(5, 6, 1100),(5, 7, 800),(6, 10, 1200),(6, 13, 2000),(7, 8, 700),(8, 9, 700),(9, 10, 900),(9, 12, 500),(9, 14, 500),(11, 12, 800),(11, 14, 800),(12, 13, 300),(13, 14, 300)]
N6 = [(1, 2, 1000),(1, 3, 1200),(2, 3, 600),(2, 4, 800),(2, 5, 1000),(3, 5, 800),(4, 5, 600),(4, 6, 1000),(5, 6, 1200)]
N6S9 = [(0,1,1000),(0,2,1200),
        (1,2,600),(1,3,800),(1,4,1000),
        (2,4,800),
        (3,4,600),(3,5,1000),
        (4,5,1200)]
RING = [(1, 2, 300), (2, 3, 230), (3, 4, 421), (4, 5, 323), (5, 6, 432), (6, 7, 272), (7, 8, 297), (8, 1, 388)]
JPN12 = [(1,2,593.3), (1, 4, 1256.4), (2, 3, 351.8), (3, 4, 47.4), 
         (3, 7, 366.0), (4, 5, 250.7), (5, 6, 252.2), (5, 7, 250.8), 
         (6, 8, 263.8), (7, 8, 185.6), (7, 10, 490.7), (8, 9, 341.6), 
         (9, 10, 66.2), (9, 11, 280.7), (10, 11, 365.0), (10, 12, 1158.7), 
         (11, 12, 911.9)]
G = nx.Graph()
G.add_weighted_edges_from(JPN12)
G = G.to_directed()


'''Graph parameters'''
V = list(G.nodes())
E = list(G.edges())
S = [1]
# s_i_set = [1 , 2 , 4]
s_i_set = [1]
Max_FS = 320
BR_M_OC = 50
#Reach_R = 4712
Reach_R = 6300
OC_Bandwidth = 37.5
Guardband = 12.5
FS_Bandwidth = 12.5
Physical_KSP_sd_set, Physical_M = KSP_ASD_and_mudulation_AP(G)

'''Gurobi ILP deviation'''
theta = 1e-5

'''Traffic volume set'''
T_volume_type = 'Random'

'''parameters of iteration'''
R_num = 100
iteration = 10
r_value1 = 100
r_value2 = 1000
Time_Limit_SLC_RMS = 30
Time_Limit_SLC_SA_S = 3600
#Time_Limit_SLC_SA_C = 3600
Time_Limit_SLC_S = 7200
Time_Limit_SLC_C = 7200


''' Simulation start'''
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(filename=os.path.join(TEST_DIR, "log", "wang.log")), 
    ]
    )
logger = logging.getLogger(__name__)

for s_i in s_i_set:
    SG = [i for i in range(1, 1 + int(len(S)/s_i))]
    obj_vals = []
    for j in range(iteration):
        R = Request_reader((j+1) * 2)
        '''Pre-calculation'''
        AP, M, P, F_rp, O_rp = Simulation_initialization()
        logger.info("R:\n%s", pformat(R))
        obj_val = ILP_SLC_Relax(Time_Limit_SLC_RMS)
        obj_vals.append(obj_val)

        # Solution_ILP_SLC_S_decomposed, Assignment_ILP_SLC_S_decomposed, undone_SLC_SA_S = ILP_SLC_decomposed_Slot(Time_Limit_SLC_SA_S)
        # Channel_bound = Find_Channel_bound(Solution_ILP_SLC_S_decomposed)
        # C, gamma = Channels_generation(F_rp, Channel_bound)
        # Solution_ILP_SLC_S, Assignment_ILP_SLC_S, undone_SLC_S = ILP_SLC_Slot(Time_Limit_SLC_S)
        # Solution_ILP_SLC_C, Assignment_ILP_SLC_C, undone_SLC_C = ILP_SLC_Channel(Time_Limit_SLC_C)

    # solutions.append(Solution_ILP_SLC_C)

    pprint(obj_vals)
