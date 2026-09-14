import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, splu

class OperatorError(RuntimeError):
    pass

MAX_CACHED_FACTORIZATIONS = 4

class Operator:
    def __init__(self, mesh, stiffness, mass, lumped, beta_scale=1.0, max_cached=None):
        self.mesh = mesh
        self.stiffness = stiffness
        self.mass = mass
        self.lumped = lumped
        self.beta_scale = beta_scale
        self.max_cached = int(MAX_CACHED_FACTORIZATIONS if max_cached is None else max_cached)
        self._cache = {}
        self._mass_factor = None

    def factor(self, ell):
        # Sparse factorization is expensive, so reuse recently evaluated ranges.
        key = round(float(ell), 9)
        cached = self._cache.pop(key, None)
        if cached is None:
            # FEM system for -div(A grad u) + ell^-2 u = q.
            system = (self.stiffness + self.mass / (ell * ell)).tocsc()
            cached = splu(system)
        self._cache[key] = cached
        while self.max_cached > 0 and len(self._cache) > self.max_cached:
            self._cache.pop(next(iter(self._cache)))
        return cached

    def solve(self, ell, load):
        return self.factor(ell).solve(np.asarray(load, dtype=float))

    def mass_solve(self, load):
        if self._mass_factor is None:
            self._mass_factor = splu(self.mass.tocsc())
        return self._mass_factor.solve(np.asarray(load, dtype=float))

    def log_range_derivative(self, ell, u):
        rhs = self.mass @ u
        return 2.0 / (ell * ell) * self.factor(ell).solve(rhs)

    def residual(self, ell, u, load):
        system = self.stiffness + self.mass / (ell * ell)
        r = system @ u - load
        scale = max(np.linalg.norm(load), 1e-300)
        return float(np.linalg.norm(r) / scale)

    def clear(self):
        self._cache = {}
        self._mass_factor = None

    def checks(self, spectrum=False, n_eigenvalues=1):

        a = self.stiffness
        difference = abs(a - a.T)
        symmetry = float(difference.max()) if difference.nnz else 0.0
        scale = float(abs(a).max()) if a.nnz else 0.0
        ones = np.ones(a.shape[0])
        row_sum = a @ ones
        constant_residual = float(np.linalg.norm(row_sum)) / max(scale, 1e-300)

        report = {
            "symmetry": symmetry,
            "relative_symmetry": symmetry / max(scale, 1e-300),
            "min_lumped_mass": float(self.lumped.min()),
            "constant_residual": constant_residual,
            "finite": bool(np.all(np.isfinite(a.data)) and np.all(np.isfinite(self.mass.data))),
            "n_components": int(self.mesh.n_components),
            "nonzeros": int(a.nnz),
            "scale": scale,
        }
        if spectrum:
            value, method = self._smallest_generalized_eigenvalue(n_eigenvalues)
            report["min_generalized_eigenvalue"] = value
            report["spectrum_method"] = method
        return report

    def _smallest_generalized_eigenvalue(self, k=1):

        a = self.stiffness
        n = a.shape[0]
        scale = float(abs(a).max()) if a.nnz else 1.0
        shift = -1e-8 * max(scale, 1e-300)
        k = int(min(max(k, 1), max(n - 2, 1)))
        if n > 3:
            try:
                values = eigsh(
                    a.tocsc(),
                    k=k,
                    M=self.mass.tocsc(),
                    sigma=shift,
                    which="LM",
                    return_eigenvectors=False,
                    maxiter=5000,
                )
                return float(np.min(values)), "shift-invert Lanczos"
            except Exception:
                pass
        rng = np.random.default_rng(0)
        worst = np.inf
        probes = [np.ones(n)] + [rng.standard_normal(n) for _ in range(8)]
        for v in probes:
            denominator = float(v @ (self.mass @ v))
            if denominator <= 0:
                continue
            worst = min(worst, float(v @ (a @ v)) / denominator)
        return (float(worst) if np.isfinite(worst) else 0.0), "Rayleigh probe"

    def validate(
        self,
        symmetry_tolerance=1e-10,
        eigenvalue_tolerance=1e-8,
        spectrum=True,
        strict=True,
    ):

        report = self.checks(spectrum=spectrum)
        failures = []

        if not report["finite"]:
            failures.append("operator contains non-finite entries")
        if report["relative_symmetry"] > symmetry_tolerance:
            failures.append(
                "stiffness asymmetry %.3e exceeds the relative tolerance %.1e"
                % (report["relative_symmetry"], symmetry_tolerance)
            )
        if report["min_lumped_mass"] <= 0:
            failures.append(
                "smallest lumped mass is %.3e; every node must carry positive mass"
                % report["min_lumped_mass"]
            )
        if report["constant_residual"] > 1e-8:
            failures.append(
                "no-flux consistency failed: ||A 1|| / scale = %.3e"
                % report["constant_residual"]
            )
        if spectrum:
            smallest = report.get("min_generalized_eigenvalue", 0.0)
            bound = -eigenvalue_tolerance * max(report["scale"], 1e-300)
            if smallest < bound:
                failures.append(
                    "smallest generalized eigenvalue %.3e is materially negative "
                    "(bound %.3e)" % (smallest, bound)
                )

        report["valid"] = not failures
        report["failures"] = failures
        if failures and strict:
            raise OperatorError(
                "transport operator failed validation: " + "; ".join(failures)
            )
        return report, failures

    def __repr__(self):
        return "Operator(%d nodes, %d nonzeros)" % (
            self.mesh.n_nodes,
            self.stiffness.nnz,
        )

def assemble(mesh, geometry, beta_scale=1.0, max_cached=None):
    # No-flux boundaries are implicit in the weak form, so no boundary term is assembled.
    nodes = mesh.nodes
    tris = mesh.triangles
    p0 = nodes[tris[:, 0]]
    p1 = nodes[tris[:, 1]]
    p2 = nodes[tris[:, 2]]

    twice_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    area = 0.5 * np.abs(twice_area)
    safe = np.where(np.abs(twice_area) < 1e-15, 1e-15, twice_area)

    bx = np.stack([p1[:, 1] - p2[:, 1], p2[:, 1] - p0[:, 1], p0[:, 1] - p1[:, 1]], axis=1) / safe[:, None]
    by = np.stack([p2[:, 0] - p1[:, 0], p0[:, 0] - p2[:, 0], p1[:, 0] - p0[:, 0]], axis=1) / safe[:, None]

    # Evaluate the H&E-derived anisotropic tensor at each triangle centroid.
    tensor = geometry.tensor_at(mesh.centroids, beta_scale=beta_scale)
    a11 = tensor[:, 0]
    a12 = tensor[:, 1]
    a22 = tensor[:, 2]

    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()

    local = np.empty((tris.shape[0], 3, 3))
    for i in range(3):
        for j in range(3):
            local[:, i, j] = area * (
                bx[:, i] * (a11 * bx[:, j] + a12 * by[:, j])
                + by[:, i] * (a12 * bx[:, j] + a22 * by[:, j])
            )
    stiffness = coo_matrix((local.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()

    # Consistent P1 mass matrix supplies the ell^-2 removal term.
    pattern = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
    mass_local = area[:, None, None] * pattern[None, :, :]
    mass = coo_matrix((mass_local.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()

    lumped = np.zeros(mesh.n_nodes)
    np.add.at(lumped, tris.ravel(), np.repeat(area / 3.0, 3))

    return Operator(
        mesh, stiffness, mass, lumped, beta_scale=beta_scale, max_cached=max_cached
    )

def exposure_field(operator, ell, load):
    u = operator.solve(ell, load)
    return u
