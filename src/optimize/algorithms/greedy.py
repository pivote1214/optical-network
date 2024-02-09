import pprint
from src.utils.graph import load_network
from src.demands.demands import gen_all_demands_offline
from src.optimize.algorithms.channel_assign import first_fit
from src.optimize.algorithms.find_paths import k_shortest_paths, k_shortest_paths_hop, kSPwLO, repeat_dijkstra
from src.optimize.algorithms.optical_network import OpticalNetwork


def greedy_RMLSA_offline(
    graph_name: str, 
    num_slots: int, 
    k: int,
    demands: dict[int, tuple[int, int, int]], 
    path_method: str, 
    alpha: float = 0.3
    ) -> OpticalNetwork:
    """Greedy RMLSA algorithm for static traffic"""
    optical_network = OpticalNetwork(graph_name, num_slots)
    # sort demands by size -> dict[int, tuple[int, int, int]]
    demands = sorted(demands.items(), key=lambda x: x[1][2], reverse=True)
    demands = {i: d[1] for i, d in enumerate(demands)}
    for d_ind in range(len(demands)):
        source, target, demand_size = demands[d_ind]
        if path_method == "kSP":
            candidate_paths = k_shortest_paths(optical_network, k, source, target)
        elif path_method == "kSP-hop":
            candidate_paths = k_shortest_paths_hop(optical_network, k, source, target)
        elif path_method == "kSPwLO":
            candidate_paths = kSPwLO(optical_network, k, alpha, source, target)
        elif path_method == "Repeat Dijkstra":
            candidate_paths = repeat_dijkstra(optical_network, k, 10, source, target)
        else:
            raise ValueError(f"path_method should be 'kSP', 'kSP-hop', or 'kSPwLO', but {path_method} is given")

        assined_slots = first_fit(optical_network, candidate_paths, demand_size)
        optical_network.renew(assined_slots)

    return optical_network


graph_name              = 'NSF'
graph                   = load_network(graph_name)
num_slots               = 320
num_demands             = 100
demands_population      = [50, 100, 150, 200]
demands_seeds_values    = [seed * 12 for seed in range(1, 11)]
k_values                = [2]
path_methods            = ["kSP", "kSP-hop", "kSPwLO"]

for path_method in path_methods:
    for k in k_values:
        print("-" * 50)
        print(f"k={k} path_method = {path_method}")
        for seed in demands_seeds_values:
            demands = gen_all_demands_offline(graph, num_demands, demands_population=demands_population, seed=seed)
            
            result_network = greedy_RMLSA_offline(graph_name, num_slots, k, demands, path_method)

            assigned_slots = result_network.occupied
            max_slots = 0
            for edge, slots in assigned_slots.items():
                for i in range(num_slots):
                    if slots[i]:
                        max_slots = max(max_slots, i + 1)
            print(max_slots)
