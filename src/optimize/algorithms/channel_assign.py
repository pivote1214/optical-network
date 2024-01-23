from src.optimize.algorithms.optical_network import OpticalNetwork

def first_fit(
    optical_network: OpticalNetwork, 
    paths_d: list[list[int]], 
    size: int
    ) -> tuple[list[int], list[int]]:
    """First fit algorithm"""
    slots_availability = optical_network.get_slots_availability(paths_d)
    