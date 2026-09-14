import numpy as np
from scipy.sparse import csc_matrix

__all__ = ["maximin_order", "neighbour_sets", "VecchiaPlan", "plan",
           "VecchiaFactor", "build_factor"]

def maximin_order(coordinates):

    points = np.atleast_2d(np.asarray(coordinates, dtype=float))
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    centroid = points.mean(axis=0)
    first = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))

    order = np.empty(n, dtype=int)
    order[0] = first
    remaining = np.ones(n, dtype=bool)
    remaining[first] = False
    best = np.sum((points - points[first]) ** 2, axis=1)
    best[first] = -np.inf
    # Spread early points across the tissue before choosing local neighbors.
    for k in range(1, n):
        candidate = int(np.argmax(np.where(remaining, best, -np.inf)))
        order[k] = candidate
        remaining[candidate] = False
        np.minimum(best, np.sum((points - points[candidate]) ** 2, axis=1), out=best)
        best[candidate] = -np.inf
    return order

def neighbour_sets(coordinates, order, n_neighbours):

    points = np.atleast_2d(np.asarray(coordinates, dtype=float))
    order = np.asarray(order, dtype=int)
    m = int(n_neighbours)
    out = [np.zeros(0, dtype=int)]
    for k in range(1, order.size):
        prefix = order[:k]
        if prefix.size <= m:
            out.append(prefix.copy())
            continue
        d = np.sum((points[prefix] - points[order[k]]) ** 2, axis=1)
        pick = np.argpartition(d, m - 1)[:m]
        out.append(prefix[pick])
    return out

class VecchiaPlan:

    def __init__(self, points, order, head_sets, tail_index, n_neighbours):
        self.points = points
        self.order = order
        self.head_sets = head_sets
        self.tail_index = tail_index
        self.n_neighbours = int(n_neighbours)

    @property
    def n(self):
        return self.points.shape[0]

def plan(coordinates, n_neighbours=30, order=None):

    points = np.atleast_2d(np.asarray(coordinates, dtype=float))
    n = points.shape[0]
    if order is None:
        order = maximin_order(points)
    order = np.asarray(order, dtype=int)
    m = int(min(n_neighbours, max(n - 1, 0)))

    head = min(m, max(n - 1, 0))
    head_sets = [order[:k] for k in range(1, head + 1)]

    start = head + 1
    tail = np.arange(start, n)
    tail_index = np.empty((tail.size, m), dtype=int)
    for row, k in enumerate(tail):
        prefix = order[:k]
        d = np.sum((points[prefix] - points[order[k]]) ** 2, axis=1)
        tail_index[row] = prefix[np.argpartition(d, m - 1)[:m]]
    return VecchiaPlan(points, order, head_sets, tail_index, m)

class VecchiaFactor:

    def __init__(self, matrix, logdet, order, n_neighbours):
        self.matrix = matrix
        self._logdet = float(logdet)
        self.order = order
        self.n_neighbours = int(n_neighbours)

    @property
    def shape(self):
        return self.matrix.shape

    def apply_transpose(self, x):
        values = np.asarray(x, dtype=float)
        flat = values.ndim == 1
        if flat:
            values = values[:, None]
        out = self.matrix.T @ values
        return out[:, 0] if flat else out

    def logdet(self):
        return self._logdet

def build_factor(coordinates, phi, nugget, kernel, order=None, n_neighbours=30):

    # Sparse conditional factors approximate Matérn whitening for large receiver sets.
    if isinstance(coordinates, VecchiaPlan):
        built = coordinates
    else:
        built = plan(coordinates, n_neighbours=n_neighbours, order=order)
    points, order, m = built.points, built.order, built.n_neighbours
    n = points.shape[0]

    tau = float(nugget)
    diagonal = tau + (1.0 - tau) * float(kernel(np.zeros(1), phi)[0])

    rows = [order[:1]]
    cols = [np.zeros(1, dtype=int)]
    data = [np.array([1.0 / np.sqrt(diagonal)])]
    logdet = np.log(diagonal)

    head = len(built.head_sets)
    for k in range(1, head + 1):
        nb = built.head_sets[k - 1]
        i = int(order[k])
        u_ii, contribution, d_k = _conditional(points, i, nb, phi, tau, kernel, diagonal)
        rows.append(np.concatenate([[i], nb]))
        cols.append(np.full(nb.size + 1, k))
        data.append(np.concatenate([[u_ii], contribution]))
        logdet += np.log(d_k)

    start = head + 1
    if start < n:
        tail = np.arange(start, n)
        nb_index = built.tail_index

        target = points[order[tail]]
        neighbour_points = points[nb_index]
        delta = neighbour_points[:, :, None, :] - neighbour_points[:, None, :, :]
        k_nn = tau * np.eye(m)[None, :, :] + (1.0 - tau) * kernel(
            np.sqrt(np.sum(delta ** 2, axis=3)), phi)
        k_ni = (1.0 - tau) * kernel(
            np.sqrt(np.sum((neighbour_points - target[:, None, :]) ** 2, axis=2)), phi)

        try:
            b = np.linalg.solve(k_nn, k_ni[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            b = np.stack([np.linalg.lstsq(k_nn[j], k_ni[j], rcond=None)[0]
                          for j in range(tail.size)])
        d_tail = diagonal - np.sum(k_ni * b, axis=1)
        d_tail = np.maximum(d_tail, 1e-12 * diagonal)
        u_ii = 1.0 / np.sqrt(d_tail)

        rows.append(order[tail])
        cols.append(tail)
        data.append(u_ii)
        rows.append(nb_index.ravel())
        cols.append(np.repeat(tail, m))
        data.append((-b * u_ii[:, None]).ravel())
        logdet += float(np.sum(np.log(d_tail)))

    matrix = csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return VecchiaFactor(matrix, logdet, order, m)

def _conditional(points, i, nb, phi, tau, kernel, diagonal):

    delta = points[nb][:, None, :] - points[nb][None, :, :]
    k_nn = tau * np.eye(nb.size) + (1.0 - tau) * kernel(
        np.sqrt(np.sum(delta ** 2, axis=2)), phi)
    k_ni = (1.0 - tau) * kernel(
        np.sqrt(np.sum((points[nb] - points[i]) ** 2, axis=1)), phi)
    try:
        b = np.linalg.solve(k_nn, k_ni)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(k_nn, k_ni, rcond=None)[0]
    d_k = max(diagonal - float(k_ni @ b), 1e-12 * diagonal)
    u_ii = 1.0 / np.sqrt(d_k)
    return u_ii, -b * u_ii, d_k
