def greedy_RMLSA_static(demands: dict[int, tuple[int, int, int]]):
    """Greedy RMLSA algorithm for static traffic"""
    for d_ind in demands:
        s, t, size = demands[d_ind]
        path_d = find_paths(s, t)
        assined_slots = first_fit(path_d, size)
        renew_network(assined_slots)


def find_paths(s: int, t: int):
    """Find paths between s and t"""
    pass

def first_fit(path_d: list[list[int]], size: int):
    """First fit algorithm"""
    pass

def renew_network(assined_slots: list[int]):
    """Renew network"""
    pass
