import numpy as np
from simplicial_complex import binary_to_simplicial_complex, plot_simplicial_complex, spatial_adjacency


class SimplicialModel:
    def __init__(self, vertices, edges, triangles, valuation):
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles
        self.valuation = valuation

    def get_adjacent_simplices(self, sigma_set, relation):
        """
        Computes Adj(Σ, C) – the set of simplices adjacent to Σ under relation C.
        """
        adjacent_simplices = set()
        for simplex1 in self.vertices | self.edges | self.triangles:
            for simplex2 in sigma_set:
                if relation({simplex1}, {simplex2}):
                    adjacent_simplices.add(simplex1)

        return adjacent_simplices

    def reach(self, phi1_simplices, phi2_simplices, relation):
        # T will hold the cumulatively discovered set of 'reachable' simplices
        T = set(phi2_simplices)

        # Frontier is the wave of newly added simplices we try to expand at each iteration
        frontier = set(phi2_simplices)

        while frontier:
            new_frontier = set()
            for sigma in frontier:
                adjacent_in_phi1 = {
                    s_adj
                    for s_adj in phi1_simplices
                    if relation({s_adj}, {sigma})
                }

                for s_adj in adjacent_in_phi1:
                    if s_adj not in T:# avoid loops
                        new_frontier.add(s_adj)
                        T.add(s_adj)

            frontier = new_frontier

        return T

    def sat(self, formula, relation):
        """
        Computes Sat(M, φ, C) recursively.
        """
        if isinstance(formula, str): # proposition
            return self.valuation.get(formula, set())

        elif formula == "⊤":  # true
            return self.vertices | self.edges | self.triangles

        elif formula[0] == "¬":  # negation
            return self.sat("⊤", relation) - self.sat(formula[1], relation)

        elif formula[0] == "∧":  # conjunction
            return self.sat(formula[1], relation) & self.sat(formula[2], relation)

        elif formula[0] == "N":  # neighborhood operator
            return self.get_adjacent_simplices(self.sat(formula[1], relation), relation)

        elif formula[0] == "R":  # reachability operator
            return self.reach(self.sat(formula[1], relation), self.sat(formula[2], relation), relation)

        else:
            raise ValueError("Unknown formula type:", formula)



binary_image = np.array([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 0]
])


vertices, edges, triangles = binary_to_simplicial_complex(binary_image)

phi1_simplices =  {((1,1),)}

phi2_simplices = {((1,0),(1,1))}
M = SimplicialModel(vertices, edges, triangles, valuation={"φ1": phi1_simplices, "φ2": phi2_simplices})
# Compute Reach(φ1, φ2, spatial adjacency)
print("Sat(Rφ1φ2, spatial adjacency):",M.sat(('R','φ1','φ2'), spatial_adjacency))


plot_simplicial_complex(vertices, edges, triangles)
