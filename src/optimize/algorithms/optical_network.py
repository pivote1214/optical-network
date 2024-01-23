import networkx as nx

from src.utils.graph import load_network


class OpticalNetwork:
    def __init__(
        self, 
        graph_name: str, 
        num_slots: int
        ) -> None:
        self.graph_name = graph_name
        self.graph = load_network(graph_name)
        self.num_slots = num_slots
        self.occupied = {(u, v): [False] * num_slots for u, v in self.graph.edges}

    def renew(self, assined_slots: tuple[list[int], list[int]]) -> None:
        """Renew network"""
        e_inds, s_inds = assined_slots
        for e_ind in e_inds:
            for s_ind in s_inds:
                self.occupied[e_ind][s_ind] = True

    def get_slots_availability(self, paths: list[list[int]]) -> dict[int, list[bool]]:
        """Get slots availability for paths"""
        slots_availability = {}
        for p_ind, path in enumerate(paths):
            slots_availability[p_ind] = [True] * self.num_slots
            for s_ind in range(self.num_slots):
                for u, v in zip(path[:-1], path[1:]):
                    if self.occupied[u, v][s_ind]:
                        slots_availability[p_ind][s_ind] = False
                        break

        return slots_availability

    
