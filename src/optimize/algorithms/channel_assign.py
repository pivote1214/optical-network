from src.optimize.algorithms.optical_network import OpticalNetwork
from src.utils.graph import calc_path_length
from src.optimize.models.RSA_PATH_CHANNEL.make_input import calc_required_slots, select_modulation_format

def first_fit(
    optical_network: OpticalNetwork, 
    candidate_paths: list[list[int]], 
    demand_size: int
    ) -> tuple[list[tuple[int, int]], list[int]]:
    """First fit algorithm"""
    # find available slots
    slots_availability = optical_network.get_slots_availability(candidate_paths)
    # first fit
    first_fit_slots = [None] * len(candidate_paths)
    for p_ind, path in enumerate(candidate_paths):
        path_length = calc_path_length(optical_network.graph, path)
        modulation_format = select_modulation_format(path_length)
        required_slots = calc_required_slots(demand_size, modulation_format, 
                                             optical_network.W, 
                                             optical_network.TRAFFIC_BPSK)
        success_slots = []
        for s_ind in range(optical_network.num_slots - required_slots + 1):
            if not slots_availability[p_ind][s_ind]:
                success_slots = []
            else:
                success_slots.append(s_ind)
                if len(success_slots) == required_slots:
                    first_fit_slots[p_ind] = success_slots
                    break

    # find first fit path
    selected_path_ind, min_ind = None, 1000
    for p_ind, path in enumerate(candidate_paths):
        if first_fit_slots[p_ind] is None:
            continue
        if first_fit_slots[p_ind][0] < min_ind:
            selected_path_ind = p_ind
            min_ind = first_fit_slots[p_ind][0]

    # if no path is selected
    if selected_path_ind is None:
        print("No path is selected")
        return None, None

    # make assigned edges and slots
    assigned_edges = []
    for u, v in zip(candidate_paths[selected_path_ind][:-1], 
                    candidate_paths[selected_path_ind][1:]):
        assigned_edges.append((u, v))
    assigned_slots = first_fit_slots[selected_path_ind]

    return assigned_edges, assigned_slots
