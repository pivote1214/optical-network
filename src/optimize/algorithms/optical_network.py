import networkx as nx

class OpticalNetwork:
    def __init__(
        self, 
        graph: nx.Graph, 
        num_slices: int
        ) -> None:
        self.graph = graph
        self.occupied = {e_ind: [False] * num_slices for e_ind in graph.edges}

    
        
        
