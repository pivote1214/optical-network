from src.utils.graph import load_network
from src.demands.demands import gen_all_demands_offline
from src.optimize.algorithms.channel_assign import first_fit
from src.optimize.algorithms.find_paths import k_shortest_paths
from src.optimize.algorithms.optical_network import OpticalNetwork


def greedy_RMLSA_offline(
    graph_name: str, 
    num_slots: int, 
    k: int,
    demands: dict[int, tuple[int, int, int]], 
    ) -> OpticalNetwork:
    """Greedy RMLSA algorithm for static traffic"""
    optical_network = OpticalNetwork(graph_name, num_slots)
    for d_ind in demands:
        source, target, demand_size = demands[d_ind]
        candidate_paths = k_shortest_paths(optical_network, k, source, target)
        assined_slots = first_fit(optical_network, candidate_paths, demand_size)
        optical_network.renew(assined_slots)

    return optical_network


graph_name              = 'NSF'
graph                   = load_network(graph_name)
num_slots               = 320
num_demands             = 500
demands_population      = [50, 100, 150, 200]
demands_seeds_values    = [seed * 12 for seed in range(1, 11)]
k_values                = [2]
# path_algo_infos         = [('kSP', None), ('kSPwLO', 0.3)]
# bound_algo              = True
# TIMELIMIT               = 3600

for seed in demands_seeds_values:
    for k in k_values:
        demands = gen_all_demands_offline(graph, num_demands, 
                                          demands_population=demands_population, 
                                          seed=seed)
        result_network = greedy_RMLSA_offline(graph_name, num_slots, k, demands)

        assigned_slots = result_network.occupied
        # pprint.pprint(assigned_slots)
        max_slots = 0
        for edge, slots in assigned_slots.items():
            for i in range(num_slots):
                if slots[i]:
                    max_slots = max(max_slots, i + 1)
        print(max_slots)
