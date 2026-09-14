import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

def contact_operator(coordinates, radius, components=None):

    coordinates = np.asarray(coordinates, float)
    tree = cKDTree(coordinates)
    neighbors = tree.query_ball_point(coordinates, r=float(radius) + 1e-9)

    rows = []
    cols = []
    for i, adjacent in enumerate(neighbors):
        for j in adjacent:
            if i == j:
                continue
            if components is not None and components[i] != components[j]:
                continue
            rows.append(i)
            cols.append(j)

    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(coordinates), len(coordinates)),
    )
    # Row normalization makes contact exposure the mean ligand score of local neighbors.
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = sparse.diags(1.0 / np.maximum(degree, 1.0))
    return (scale @ adjacency).tocsr()

def contact_exposure(coordinates, ligand_score, median_nn, components=None):

    # Contact-annotated interactions use neighbors within 1.5 times the median nearest-neighbor distance.
    operator = contact_operator(coordinates, 1.5 * float(median_nn), components)
    return np.asarray(operator @ np.asarray(ligand_score, float)).ravel()
