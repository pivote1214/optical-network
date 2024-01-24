import networkx as nx
from dataclasses import field

from src.utils.graph import load_network


class OpticalNetwork:
    def __init__(
        self, 
        graph_name: str, 
        num_slots: int, 
        W: dict[str, float] = {"OC": 37.5, "GB": 6.25, "FS": 12.5}, 
        TRAFFIC_BPSK: float = 50
        ) -> None:
        self.graph_name = graph_name
        self.graph = load_network(graph_name)
        self.num_slots = num_slots
        self.occupied = {(u, v): [False] * num_slots for u, v in self.graph.edges}
        self.W = W
        self.TRAFFIC_BPSK = TRAFFIC_BPSK

    def renew(self, assined_slots: tuple[list[tuple[int, int]], list[int]]) -> None:
        """Renew network"""
        edges, s_inds = assined_slots
        for u, v in edges:
            for s_ind in s_inds:
                self.occupied[u, v][s_ind] = True

    def get_slots_availability(self, candidate_paths: list[list[int]]) -> dict[int, list[bool]]:
        """Get slots availability for candidate_paths"""
        slots_availability = [None] * len(candidate_paths)
        for p_ind, path in enumerate(candidate_paths):
            slots_availability[p_ind] = [True] * self.num_slots
            for s_ind in range(self.num_slots):
                for u, v in zip(path[:-1], path[1:]):
                    if self.occupied[u, v][s_ind]:
                        slots_availability[p_ind][s_ind] = False
                        break

        return slots_availability

    
