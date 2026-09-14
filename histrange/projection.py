import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .mesh import OutsideTissueError

def _validated(mesh, coordinates, boundary_tolerance_um, label, location=None):

    coords = np.atleast_2d(np.asarray(coordinates, dtype=float))
    if location is None:
        tri, weights, report = mesh.require_inside(
            coords, boundary_tolerance_um=boundary_tolerance_um, label=label
        )
        return coords, tri, weights, report

    tri, weights = location
    tri = np.asarray(tri, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if tri.shape[0] != coords.shape[0] or weights.shape != (coords.shape[0], 3):
        raise ValueError("precomputed location does not match the supplied coordinates")
    outside = np.flatnonzero(tri < 0)
    if outside.size:
        preview = ", ".join(str(int(i)) for i in outside[:10])
        raise OutsideTissueError(
            "%d of %d %s lie outside the meshed tissue support: indices %s"
            % (outside.size, tri.size, label, preview)
        )
    return coords, tri, weights, None

def _point_rows(mesh, tri, weights, k):
    nodes = mesh.triangles[tri[k]]
    out = []
    for slot in range(3):
        if weights[k, slot] > 0:
            out.append((int(nodes[slot]), float(weights[k, slot])))
    return out

def _footprint_groups(mesh, coords, tri, radius_um):

    tree = cKDTree(mesh.nodes)
    groups = tree.query_ball_point(coords, r=float(radius_um))
    own = mesh.node_components[mesh.triangles[tri, 0]]
    cleaned = []
    for k, group in enumerate(groups):
        if len(group) == 0:
            cleaned.append(np.zeros(0, dtype=int))
            continue
        g = np.asarray(group, dtype=int)
        cleaned.append(g[mesh.node_components[g] == own[k]])
    return cleaned

def source_operator(
    mesh, coordinates, radius_um=None, boundary_tolerance_um=0.0, location=None
):
    coords, tri, weights, _ = _validated(
        mesh, coordinates, boundary_tolerance_um, "source observations", location
    )
    n_obs = coords.shape[0]
    rows, cols, data = [], [], []

    if radius_um is None or radius_um <= 0:
        for k in range(n_obs):
            for node, weight in _point_rows(mesh, tri, weights, k):
                rows.append(node)
                cols.append(k)
                data.append(weight)
    else:
        groups = _footprint_groups(mesh, coords, tri, radius_um)
        for k, group in enumerate(groups):
            if group.size == 0:
                for node, weight in _point_rows(mesh, tri, weights, k):
                    rows.append(node)
                    cols.append(k)
                    data.append(weight)
                continue
            # Distribute each source observation across its physical capture footprint.
            w = mesh.node_areas[group]
            total = w.sum()
            if total <= 0:
                w = np.ones(group.size)
                total = float(group.size)
            w = w / total
            rows.extend(group.tolist())
            cols.extend([k] * group.size)
            data.extend(w.tolist())

    return csr_matrix((data, (rows, cols)), shape=(mesh.n_nodes, n_obs))

def receiver_operator(
    mesh, coordinates, radius_um=None, boundary_tolerance_um=0.0, location=None
):
    coords, tri, weights, _ = _validated(
        mesh, coordinates, boundary_tolerance_um, "receiver observations", location
    )
    n_obs = coords.shape[0]
    rows, cols, data = [], [], []

    if radius_um is None or radius_um <= 0:
        for k in range(n_obs):
            for node, weight in _point_rows(mesh, tri, weights, k):
                rows.append(k)
                cols.append(node)
                data.append(weight)
    else:
        groups = _footprint_groups(mesh, coords, tri, radius_um)
        for k, group in enumerate(groups):
            if group.size == 0:
                for node, weight in _point_rows(mesh, tri, weights, k):
                    rows.append(k)
                    cols.append(node)
                    data.append(weight)
                continue
            # Average the solved field over the receiver capture footprint.
            w = mesh.node_areas[group]
            total = w.sum()
            if total <= 0:
                w = np.ones(group.size)
                total = float(group.size)
            w = w / total
            rows.extend([k] * group.size)
            cols.extend(group.tolist())
            data.extend(w.tolist())

    return csr_matrix((data, (rows, cols)), shape=(n_obs, mesh.n_nodes))

def deposit(source_matrix, ligand):
    return np.asarray(source_matrix @ np.asarray(ligand, dtype=float)).ravel()

def nodal_density(load, node_areas):
    areas = np.where(node_areas > 0, node_areas, 1.0)
    return np.asarray(load, dtype=float) / areas

def interpolate(receiver_matrix, field):
    return np.asarray(receiver_matrix @ np.asarray(field, dtype=float)).ravel()
