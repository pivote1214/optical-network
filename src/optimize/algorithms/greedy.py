def greedy_RMLSA_static(demands: dict[int, tuple[int, int, int]]):
    """Greedy RMLSA algorithm for static traffic"""
    for d_ind, demand in demands.items():
        source, destination, traffic = demand
