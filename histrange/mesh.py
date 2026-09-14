import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .settings import DEFAULT_MESH_UM, FALLBACK_MESH_UM, MAX_MESH_NODES

class OutsideTissueError(ValueError):
    pass

class Mesh:
    def __init__(self, nodes, triangles, spacing_um):
        self.nodes = nodes
        self.triangles = triangles
        self.spacing_um = float(spacing_um)
        self._tree = None
        self._geodesic = None
        self._component_diameters = None
        p0 = nodes[triangles[:, 0]]
        p1 = nodes[triangles[:, 1]]
        p2 = nodes[triangles[:, 2]]
        self.centroids = (p0 + p1 + p2) / 3.0
        cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
            p1[:, 1] - p0[:, 1]
        )
        self.areas = 0.5 * np.abs(cross)
        self.node_areas = np.zeros(len(nodes))
        np.add.at(self.node_areas, triangles.ravel(), np.repeat(self.areas / 3.0, 3))
        self.n_components, self.node_components = _components(nodes.shape[0], triangles)

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_triangles(self):
        return self.triangles.shape[0]

    @property
    def connected(self):
        return self.n_components == 1

    def component_areas(self):
        areas = np.zeros(self.n_components)
        labels = self.node_components[self.triangles[:, 0]]
        np.add.at(areas, labels, self.areas)
        return areas

    def edges(self):
        t = self.triangles
        pairs = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        return np.unique(np.sort(pairs, axis=1), axis=0)

    def graph(self):
        e = self.edges()
        length = np.linalg.norm(self.nodes[e[:, 0]] - self.nodes[e[:, 1]], axis=1)
        n = self.n_nodes
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        data = np.concatenate([length, length])
        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    def component_diameters(self):

        if self._component_diameters is None:
            g = self.graph()
            out = np.zeros(self.n_components)
            for label in range(self.n_components):
                members = np.flatnonzero(self.node_components == label)
                if members.size < 2:
                    continue
                start = int(members[0])
                d = dijkstra(g, indices=start)
                d = np.where(np.isfinite(d), d, -np.inf)
                far = int(np.argmax(d))
                d2 = dijkstra(g, indices=far)
                d2 = np.where(np.isfinite(d2), d2, -np.inf)
                out[label] = float(np.max(d2))
            self._component_diameters = out
        return self._component_diameters

    def geodesic_diameter(self):

        if self._geodesic is None:
            diameters = self.component_diameters()
            self._geodesic = float(np.max(diameters)) if diameters.size else 0.0
        return self._geodesic

    def median_nearest_neighbour(self):
        e = self.edges()
        length = np.linalg.norm(self.nodes[e[:, 0]] - self.nodes[e[:, 1]], axis=1)
        return float(np.median(length))

    def locate(self, points, n_candidates=16, boundary_tolerance_um=0.0):

        pts = np.atleast_2d(np.asarray(points, dtype=float))
        n = pts.shape[0]
        tri = np.full(n, -1, dtype=int)
        weights = np.zeros((n, 3))
        snapped = np.zeros(n, dtype=bool)
        distance = np.zeros(n)

        if n == 0:
            return tri, weights, _location_report(tri, snapped, distance)

        if self._tree is None:
            self._tree = cKDTree(self.centroids)

        k = int(min(n_candidates, self.n_triangles))
        _, candidates = self._tree.query(pts, k=k)
        candidates = np.asarray(candidates)
        if candidates.ndim == 1:
            candidates = candidates.reshape(n, 1) if n > 1 else candidates.reshape(1, -1)
        if candidates.shape[0] != n:
            candidates = candidates.T

        pending = np.arange(n)
        for column in range(candidates.shape[1]):
            if pending.size == 0:
                break
            idx = candidates[pending, column]
            bary = self._barycentric(pts[pending], idx)
            good = bary.min(axis=1) >= -1e-9
            hit = pending[good]
            if hit.size:
                tri[hit] = idx[good]
                w = np.clip(bary[good], 0.0, None)
                weights[hit] = w / w.sum(axis=1, keepdims=True)
            pending = pending[~good]

        if pending.size and boundary_tolerance_um > 0:
            recovered, w, d = self._clamp_to_nearest(pts[pending], candidates[pending])
            within = d <= float(boundary_tolerance_um)
            hit = pending[within]
            if hit.size:
                tri[hit] = recovered[within]
                weights[hit] = w[within]
                snapped[hit] = True
                distance[hit] = d[within]
            pending = pending[~within]

        return tri, weights, _location_report(tri, snapped, distance)

    def contains(self, points, boundary_tolerance_um=0.0):
        tri, _, _ = self.locate(points, boundary_tolerance_um=boundary_tolerance_um)
        return tri >= 0

    def require_inside(self, points, boundary_tolerance_um=0.0, label="observations"):

        tri, weights, report = self.locate(
            points, boundary_tolerance_um=boundary_tolerance_um
        )
        outside = np.flatnonzero(tri < 0)
        if outside.size:
            preview = ", ".join(str(int(i)) for i in outside[:10])
            more = "" if outside.size <= 10 else " (and %d more)" % (outside.size - 10)
            raise OutsideTissueError(
                "%d of %d %s lie outside the meshed tissue support or inside a hole: "
                "indices %s%s. Filter them out before fitting; they must not be "
                "projected onto tissue." % (outside.size, tri.size, label, preview, more)
            )
        return tri, weights, report

    def component_of(self, points, boundary_tolerance_um=0.0):

        tri, _, _ = self.locate(points, boundary_tolerance_um=boundary_tolerance_um)
        labels = np.full(tri.shape[0], -1, dtype=int)
        inside = tri >= 0
        labels[inside] = self.node_components[self.triangles[tri[inside], 0]]
        return labels

    def _clamp_to_nearest(self, pts, candidates):

        best_tri = np.full(pts.shape[0], -1, dtype=int)
        best_w = np.zeros((pts.shape[0], 3))
        best_d = np.full(pts.shape[0], np.inf)
        for column in range(candidates.shape[1]):
            idx = candidates[:, column]
            bary = self._barycentric(pts, idx)
            w = np.clip(bary, 0.0, None)
            total = w.sum(axis=1, keepdims=True)
            total = np.where(total <= 0, 1.0, total)
            w = w / total
            a = self.nodes[self.triangles[idx, 0]]
            b = self.nodes[self.triangles[idx, 1]]
            c = self.nodes[self.triangles[idx, 2]]
            position = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3] * c
            d = np.linalg.norm(position - pts, axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_tri[better] = idx[better]
            best_w[better] = w[better]
        return best_tri, best_w, best_d

    def _barycentric(self, pts, tri_index):
        a = self.nodes[self.triangles[tri_index, 0]]
        b = self.nodes[self.triangles[tri_index, 1]]
        c = self.nodes[self.triangles[tri_index, 2]]
        v0 = b - a
        v1 = c - a
        v2 = pts - a
        det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / det
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / det
        return np.stack([1.0 - l1 - l2, l1, l2], axis=1)

    def __repr__(self):
        return "Mesh(%d nodes, %d triangles, h=%.1f um, %d component%s)" % (
            self.n_nodes,
            self.n_triangles,
            self.spacing_um,
            self.n_components,
            "" if self.n_components == 1 else "s",
        )

def _location_report(tri, snapped, distance):
    inside = tri >= 0
    return {
        "n_points": int(tri.size),
        "n_inside": int(np.count_nonzero(inside)),
        "n_outside": int(np.count_nonzero(~inside)),
        "outside_indices": np.flatnonzero(~inside),
        "n_snapped": int(np.count_nonzero(snapped)),
        "snapped_indices": np.flatnonzero(snapped),
        "max_snap_distance_um": float(distance.max()) if distance.size else 0.0,
    }

def _components(n_nodes, triangles):
    pairs = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    data = np.ones(pairs.shape[0])
    graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
    count, labels = connected_components(graph, directed=False)
    return int(count), labels

def build(histology, spacing_um=None, keep_components="all"):

    if keep_components not in ("all", "largest"):
        raise ValueError("keep_components must be 'all' or 'largest'")
    if spacing_um is not None:
        candidates = [float(spacing_um)]
    else:
        # Prefer the 4 um mesh and use coarser fallbacks only when the tissue is too large.
        candidates = [DEFAULT_MESH_UM] + list(FALLBACK_MESH_UM)
    max_nodes = MAX_MESH_NODES

    failure = None
    for h in candidates:
        try:
            return _lattice_mesh(histology, h, max_nodes, keep_components)
        except ValueError as exc:
            failure = exc
            continue
    if failure is not None:
        raise failure
    raise ValueError("no admissible mesh spacing")

def _lattice_mesh(histology, h, max_nodes, keep_components):
    mask = histology.mask
    px = histology.pixel_size_um
    rows, cols = np.nonzero(mask)
    x0 = cols.min() * px
    x1 = cols.max() * px
    y0 = rows.min() * px
    y1 = rows.max() * px

    xs = np.arange(x0, x1 + 0.5 * h, h)
    ys = np.arange(y0, y1 + 0.5 * h, h)
    if xs.size < 2 or ys.size < 2:
        raise ValueError("tissue is too small for mesh spacing %.1f um" % h)

    gx, gy = np.meshgrid(xs, ys)
    ci = np.rint(gx / px).astype(int)
    ri = np.rint(gy / px).astype(int)
    valid = (ci >= 0) & (ci < mask.shape[1]) & (ri >= 0) & (ri < mask.shape[0])
    inside = np.zeros(valid.shape, dtype=bool)
    inside[valid] = mask[ri[valid], ci[valid]]

    if inside.sum() > max_nodes:
        raise ValueError("mesh at %.1f um would exceed the node budget" % h)

    index = np.arange(inside.size).reshape(inside.shape)
    full = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]
    jj, ii = np.nonzero(full)
    if jj.size == 0:
        raise ValueError("no full cells at mesh spacing %.1f um" % h)

    n00 = index[jj, ii]
    n10 = index[jj, ii + 1]
    n01 = index[jj + 1, ii]
    n11 = index[jj + 1, ii + 1]

    # Alternate cell diagonals to avoid directional mesh bias.
    flip = (ii + jj) % 2 == 0
    tri_a = np.where(
        flip[:, None], np.stack([n00, n10, n11], axis=1), np.stack([n00, n10, n01], axis=1)
    )
    tri_b = np.where(
        flip[:, None], np.stack([n00, n11, n01], axis=1), np.stack([n10, n11, n01], axis=1)
    )
    triangles = np.vstack([tri_a, tri_b])

    used, triangles = np.unique(triangles, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    nodes = np.stack([gx.ravel()[used], gy.ravel()[used]], axis=1)

    if keep_components == "largest":
        nodes, triangles = _keep_largest_component(nodes, triangles)

    mesh = Mesh(nodes, triangles, h)
    _orient(mesh)
    if keep_components == "all" and mesh.n_components > 1:
        areas = mesh.component_areas()
        warnings.warn(
            "tissue mesh has %d disconnected components (areas um^2: %s); all are "
            "retained, but transport cannot cross between them"
            % (mesh.n_components, np.array2string(np.sort(areas)[::-1][:5], precision=0)),
            stacklevel=3,
        )
    return mesh

def _keep_largest_component(nodes, triangles):
    n = nodes.shape[0]
    count, labels = _components(n, triangles)
    if count == 1:
        return nodes, triangles
    sizes = np.bincount(labels)
    keep = int(np.argmax(sizes))
    dropped = int(n - sizes[keep])
    warnings.warn(
        "keep_components='largest' discards %d of %d mesh nodes in %d smaller tissue "
        "components" % (dropped, n, count - 1),
        stacklevel=3,
    )
    node_keep = labels == keep
    remap = -np.ones(n, dtype=int)
    remap[node_keep] = np.arange(node_keep.sum())
    tri_keep = node_keep[triangles].all(axis=1)
    return nodes[node_keep], remap[triangles[tri_keep]]

def _orient(mesh):
    p0 = mesh.nodes[mesh.triangles[:, 0]]
    p1 = mesh.nodes[mesh.triangles[:, 1]]
    p2 = mesh.nodes[mesh.triangles[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    flip = cross < 0
    if flip.any():
        mesh.triangles[flip] = mesh.triangles[flip][:, [0, 2, 1]]
