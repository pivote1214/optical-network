import pickle
import networkx as nx

from src.utils.paths import GRAPH_DIR


N6 = [(1, 2, 1000), (1, 3, 1200), 
      (2, 3, 600), (2, 4, 800), (2, 5, 1000), 
      (3, 5, 800), 
      (4, 5, 600), (4, 6, 1000), 
      (5, 6, 1200)]

N6S9 = [(0, 1, 1000), (0, 2, 1200), 
        (1, 2, 600), (1, 3, 800), (1, 4, 1000), 
        (2, 4, 800), 
        (3, 4, 600), (3, 5, 1000), 
        (4, 5, 1200)]

RING = [(1, 2, 300), 
        (2, 3, 230), 
        (3, 4, 421), 
        (4, 5, 323), 
        (5, 6, 432), 
        (6, 7, 272), 
        (7, 8, 297), 
        (8, 1, 388)]

NSF = [(1, 2, 1100), (1, 3, 1600), (1, 8, 2800), 
       (2, 3, 600), (2, 4, 1000), 
       (3, 6, 2000), 
       (4, 5, 600), (4, 11, 2400), 
       (5, 6, 1100), (5, 7, 800), 
       (6, 10, 1200), (6, 13, 2000), 
       (7, 8, 700), 
       (8, 9, 700), 
       (9, 10, 900), (9, 12, 500), (9, 14, 500), 
       (11, 12, 800), (11, 14, 800), 
       (12, 13, 300), 
       (13, 14, 300)]

EURO16 = [(1, 2, 514), (1, 4, 540), 
          (2, 3, 393), (2, 5, 594), (2, 7, 600), 
          (3, 4, 259), (3, 9, 474), 
          (4, 8, 552), 
          (5, 6, 507), 
          (6, 7, 218), (6, 10, 327), 
          (7, 9, 271), 
          (8, 9, 592), (8, 12, 381), 
          (9, 11, 456), 
          (10, 11, 522), (10, 13, 720), 
          (11, 12, 757), (11, 15, 534), 
          (12, 16, 420), 
          (13, 14, 783), 
          (14, 15, 400), 
          (15, 16, 376)]

JPN25 = [(1, 2, 593.3), (1, 12, 931.1), 
        (2, 3, 79), (2, 4, 245.4), 
        (3, 5, 163.3), (3, 12, 180.1), 
        (4, 5, 95.6), (4, 7, 117), (4, 8, 127.5), 
        (5, 6, 106.5), (5, 7, 79.2), 
        (6, 7, 74.7), (6, 10, 96.4), (6, 12, 228.9), (6, 14, 117.4), 
        (7, 8, 66.1), (7, 9, 30.3), 
        (8, 9, 39.2), 
        (9, 10, 47.4), (9, 11, 28.8), 
        (10, 11, 36.5), (10, 14, 250.7), 
        (11, 16, 151.4), 
        (12, 14, 211.3), 
        (13, 14, 252.2), (13, 18, 224.8), 
        (14, 17, 250.8), 
        (15, 17, 30.3), (15, 18, 117.3), 
        (16, 17, 185.8), 
        (17, 19, 208), 
        (18, 19, 39), (18, 20, 77.4), 
        (19, 20, 36.9), (19, 22, 410.8), 
        (20, 21, 304.7), 
        (21, 22, 66.2), (21, 23, 280.7), 
        (22, 23, 365), (22, 24, 314.5), 
        (23, 24, 118.4), (23, 25, 911.9), 
        (24, 25, 844.2)]

US24 = [(0, 1, 800), (0, 5, 1000), 
        (1, 2, 1100), (1, 5, 950), 
        (2, 3, 250), (2, 4, 960), (2, 6, 1000), 
        (3, 4, 800), (3, 6, 850), 
        (4, 7, 1200), 
        (5, 6, 1000), (5, 8, 1200), (5, 10, 1900), 
        (6, 7, 1150), (6, 8, 1000), 
        (7, 9, 900), 
        (8, 9, 1000), (8, 10, 1400), (8, 11, 1000), 
        (9, 12, 950), (9, 13, 850), 
        (10, 11, 900), (10, 14, 1300), (10, 18, 2600), 
        (11, 12, 900), (11, 15, 1000), 
        (12, 13, 650), (12, 16, 1100), 
        (13, 17, 1200), 
        (14, 15, 600), (14, 19, 1300), 
        (15, 16, 1000), (15, 20, 1000), (15, 21, 800), 
        (16, 17, 800), (16, 21, 850), (16, 22, 1000), 
        (17, 23, 900), 
        (18, 19, 1200), 
        (19, 20, 700), 
        (20, 21, 300), 
        (21, 22, 600), 
        (22, 23, 900)]

JPN12 = {(1, 2, 593.3), (1, 4, 1256.4), 
        (2, 3, 351.8), 
        (3, 4, 47.4), (3, 7, 366.0), 
        (4, 5, 250.7), 
        (5, 6, 252.2), (5, 7, 260.8), 
        (6, 8, 263.8), 
        (7, 8, 185.6), (7, 10, 490.7), 
        (8, 9, 341.6), 
        (9, 10, 66.2), (9, 11, 280.7), 
        (10, 11, 365.0), (10, 12, 1158.7), 
        (11, 12, 911.9)}


cr_table_network = {'NSF': NSF, 
                    'N6': N6, 
                    'N6S9': N6S9, 
                    'RING': RING, 
                    'EURO16': EURO16, 
                    'US24': US24, 
                    'JPN12': JPN12, 
                    'JPN25': JPN25}


def load_network(network_name: str) -> nx.DiGraph:
    """load graph from pickle file"""
    full_path = GRAPH_DIR / f"{network_name}.pickle"
    with open(full_path, 'rb') as f:
        graph = pickle.load(f)
        
    return graph


def create_network(network_name: str) -> nx.DiGraph:
    """
    create graph from network name
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(cr_table_network[network_name])
    # convert to directed graph
    graph = nx.to_directed(graph)

    return graph


def calc_path_length(graph: nx.DiGraph, path: list[int]) -> int:
    """calculate path length"""
    path_length = 0
    for i in range(len(path) - 1):
        path_length += graph[path[i]][path[i + 1]]['weight']

    return path_length


def calc_path_similarity(
    graph: nx.DiGraph, 
    path1: list[int], 
    path2: list[int]
    ) -> float:
    """calculate path similarity"""
    edges_path1 = {(path1[i], path1[i + 1]) for i in range(len(path1) - 1)}
    edges_path2 = {(path2[i], path2[i + 1]) for i in range(len(path2) - 1)}

    common_edges = sum([graph[a][b]['weight'] for a, b in edges_path1 & edges_path2])
    path1_edges = sum([graph[a][b]['weight'] for a, b in edges_path1])
    path2_edges = sum([graph[a][b]['weight'] for a, b in edges_path2])

    if min(path1_edges, path2_edges) == 0:
        similarity = 0
    else:
        similarity = common_edges / min(path1_edges, path2_edges)

    return similarity


def is_edge_in_path(path: list[int], edge: tuple[int, int]) -> bool:
    """judge whether edge is in path"""
    judge = False
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) == edge:
            judge = True

    return judge


def judge_common_edges(path1: list[int], path2: list[int]) -> bool:
    """judge whether two paths have common edges"""
    judge = False
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            if (path1[i], path1[i + 1]) == (path2[j], path2[j + 1]):
                judge = True

    return judge


# # make graph pickle file
# if __name__ == "__main__":
#     print('Enter network name: ')
#     print('Candidate: N6, N6S9, RING, NSF, EURO16, US24, JPN12, JPN25')
#     target_network = list(input('Ex) NSF, EURO16:\n').split(', '))
#     print('Creating graph pickle file...')
#     for network_name in target_network:
#         graph = create_network(network_name)
#         for u in graph.nodes:
#             for v in graph.nodes:
#                 if u != v and nx.shortest_path_length(graph, u, v, weight='weight') > 6300:
#                     print('Too long path!')
#                     print('network_name: ', network_name)
#                     print(u, v, nx.shortest_path_length(graph, u, v, weight='weight'))
                        
#         full_path = GRAPH_DIR / f"{network_name}.pickle"
#         with open(full_path, 'wb') as f:
#             pickle.dump(graph, f)
