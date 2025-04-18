import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# PROPERTIES
def is_background():
    pass


def is_tumor_():
    pass


def is_surrounded_by_brain():
    pass


def is_contiguous():
    pass


def has_proper_volume():
    pass

def has_acceptable_tumor_count():
    pass


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_form_x(edge1, edge2):
    a, b = edge1
    c, d = edge2
    return ccw(a, b, c) != ccw(a, b, d) and ccw(c, d, a) != ccw(c, d, b) and a not in edge2 and b not in edge2


def binary_to_simplicial_complex(binary_image):
    # 0-simplices: vertices (store each as a single-item tuple)
    rows, cols = np.where(binary_image == 1)
    vertices = {((int(r), int(c)),) for r, c in zip(rows, cols)}

    # 1-simplices: edges (2-tuples of plain (x,y) points)
    edges = set()
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for v in vertices:
        (x, y) = v[0]
        for (dx, dy) in neighbor_offsets:
            neighbor = (x + dx, y + dy)
            if (neighbor,) in vertices:
                new_edge = tuple(sorted([(x, y), neighbor]))

                if not any(segments_form_x(new_edge, existing_edge) for existing_edge in edges):
                    edges.add(new_edge)

    # 2-simplices: triangles (3-tuples of plain (x,y) points)
    triangles = set()

    for (a, b) in edges:
        for c in vertices:
            c_coord = c[0]
            if c_coord != a and c_coord != b:
                edge_ac = tuple(sorted([a, c_coord]))
                edge_bc = tuple(sorted([b, c_coord]))
                if edge_ac in edges and edge_bc in edges:
                    triangle = tuple(sorted([a, b, c_coord]))
                    triangles.add(triangle)

    return vertices, edges, triangles


def plot_simplicial_complex(vertices, edges, triangles):
    fig, ax = plt.subplots()

    for tri in triangles:
        # convert each (row, col) -> (col, row) for plotting:
        tri_coords = [(p[1], p[0]) for p in tri]
        polygon = Polygon(tri_coords,
                          closed=True,
                          facecolor=np.random.rand(3, ),
                          alpha=0.3,
                          edgecolor='none')
        ax.add_patch(polygon)

    for edge in edges:
        (x1, y1), (x2, y2) = edge
        ax.plot([y1, y2],
                [x1, x2],
                color='black', linewidth=1)

    for vtx_tuple in vertices:
        (x, y) = vtx_tuple[0]
        ax.plot(y, x, 'ro', markersize=3)

    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def lower_adjacency(sigma1, sigma2):
    pass


def upper_adjacency(sigma1, sigma2):
    pass


def spatial_adjacency(sigma1, sigma2):
    for s1 in sigma1:
        for s2 in sigma2:
            if set(s1) & set(s2):
                return True
    return False
