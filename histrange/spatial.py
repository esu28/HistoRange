import numpy as np
from scipy import optimize
from scipy.linalg import cholesky, solve_triangular, svd
from scipy.spatial.distance import cdist

from . import vecchia as vecchia_module
from .settings import (DENSE_GP_MAX_RECEIVERS, NUGGET_BOUNDS, NUGGET_START, VECCHIA_NEIGHBORS, FALLBACK_PHI_POINTS, FALLBACK_NUGGET_GRID, SECRETED_CALIBRATION_C, SECRETED_CALIBRATION_GAMMA, BOOTSTRAP_DRAWS)

RCOND = 1e-10

class Covariance:
    def __init__(self, phi, nugget, sigma2, chol, converged, fit_method="manual",
                 approximation="dense", factor=None):
        self.phi = phi
        self.nugget = nugget
        self.sigma2 = sigma2
        self.chol = chol
        self.converged = bool(converged)
        self.fit_method = str(fit_method)
        self.approximation = str(approximation)
        self.factor = factor

    def whiten(self, x):
        if self.factor is None:
            return solve_triangular(self.chol, np.asarray(x, dtype=float), lower=True)
        return self.factor.apply_transpose(np.asarray(x, dtype=float)) / np.sqrt(self.sigma2)

    def quadratic(self, x):
        w = self.whiten(x)
        return float(np.sum(w * w))

    def __repr__(self):
        return (
            "Covariance(phi=%.1f um, nugget=%.2f, converged=%s, fit_method=%s, "
            "approximation=%s)"
            % (self.phi, self.nugget, self.converged, self.fit_method, self.approximation)
        )

def matern32(distance, phi):
    scaled = np.sqrt(3.0) * distance / phi
    return (1.0 + scaled) * np.exp(-scaled)

def orthonormal_basis(matrix, rcond=RCOND):

    a = np.atleast_2d(np.asarray(matrix, dtype=float))
    if a.size == 0:
        return np.zeros((a.shape[0], 0)), 0, np.zeros(0)
    u, s, _ = svd(a, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((a.shape[0], 0)), 0, s
    rank = int(np.sum(s > rcond * s[0]))
    return u[:, :rank], rank, s

def gls_fit(design, response, rcond=RCOND):

    a = np.atleast_2d(np.asarray(design, dtype=float))
    y = np.asarray(response, dtype=float).ravel()
    u, s, vt = svd(a, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return {
            "coefficients": np.zeros(a.shape[1]),
            "rss": float(np.sum(y * y)),
            "basis": np.zeros((a.shape[0], 0)),
            "rank": 0,
            "full_rank": a.shape[1] == 0,
            "condition": np.inf,
        }
    rank = int(np.sum(s > rcond * s[0]))
    ur = u[:, :rank]
    inv_s = np.zeros_like(s)
    inv_s[:rank] = 1.0 / s[:rank]
    coefficients = vt.T @ (inv_s * (u.T @ y))
    residual = y - ur @ (ur.T @ y)
    condition = float(s[0] / s[rank - 1]) if rank > 0 else np.inf
    return {
        "coefficients": coefficients,
        "rss": float(np.sum(residual * residual)),
        "basis": ur,
        "rank": rank,
        "full_rank": rank == a.shape[1],
        "condition": condition,
    }

def fit_covariance(response, baseline, coordinates, phi_bounds, method="auto", dense_max_receivers=None):

    # Fit residual spatial covariance under the baseline-only model.
    y = np.asarray(response, dtype=float)
    b = np.asarray(baseline, dtype=float)
    n = b.shape[0]
    coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))

    if dense_max_receivers is None:
        dense_max_receivers = DENSE_GP_MAX_RECEIVERS
    if method == "auto":
        # Large receiver sets use a nearest-neighbor GP approximation.
        method = "dense" if n <= int(dense_max_receivers) else "vecchia"
    if method not in ("dense", "vecchia"):
        raise ValueError("method must be 'auto', 'dense' or 'vecchia'")

    lo_phi, hi_phi = phi_bounds
    lo_tau, hi_tau = NUGGET_BOUNDS
    start_tau = NUGGET_START

    if method == "vecchia":
        neighbours = VECCHIA_NEIGHBORS
        ordering = "max-min"
        if ordering not in ("max-min", "maxmin"):
            raise ValueError(
                "only the configured 'max-min' Vecchia ordering is implemented, got %r" % ordering
            )
        vecchia_plan = vecchia_module.plan(coordinates, n_neighbours=neighbours)

        def whitened(phi, tau):
            factor = vecchia_module.build_factor(vecchia_plan, phi, tau, matern32)
            return factor.apply_transpose(b), factor.apply_transpose(y), factor.logdet(), factor
    else:
        distance = cdist(coordinates, coordinates)

        def build(phi, tau):
            k = matern32(distance, phi)
            return tau * np.eye(n) + (1.0 - tau) * k

        def whitened(phi, tau):
            lv = cholesky(build(phi, tau), lower=True)
            logdet_v = 2.0 * float(np.sum(np.log(np.diag(lv))))
            return (
                solve_triangular(lv, b, lower=True),
                solve_triangular(lv, y, lower=True),
                logdet_v,
                lv,
            )

    def objective(params):
        phi = float(np.exp(params[0]))
        tau = float(params[1])
        try:
            bw, yw, logdet_v, _ = whitened(phi, tau)
        except Exception:
            return 1e12
        basis, rank, s = orthonormal_basis(bw)
        if rank == 0:
            return 1e12
        resid = yw - basis @ (basis.T @ yw)
        rss = float(np.sum(resid * resid))
        if rss <= 0 or n <= rank:
            return 1e12
        s2 = rss / (n - rank)
        logdet_b = 2.0 * float(np.sum(np.log(s[:rank])))
        return (n - rank) * np.log(s2) + logdet_v + logdet_b

    starts = []
    fractions = [0.15, 0.35, 0.5, 0.7, 0.9]
    taus = [start_tau, 0.10, 0.35, 0.60, 0.85]
    for f, t in zip(fractions, taus):
        phi = lo_phi * (hi_phi / lo_phi) ** f
        starts.append([np.log(phi), min(max(t, lo_tau), hi_tau)])

    best = None
    for start in starts:
        try:
            result = optimize.minimize(
                objective,
                np.array(start),
                method="L-BFGS-B",
                bounds=[(np.log(lo_phi), np.log(hi_phi)), (lo_tau, hi_tau)],
            )
        except Exception:
            continue
        if not bool(getattr(result, "success", False)):
            continue
        if not np.isfinite(result.fun) or result.fun >= 1e11:
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        grid_phi = np.geomspace(
            lo_phi, hi_phi, FALLBACK_PHI_POINTS
        )
        grid_tau = FALLBACK_NUGGET_GRID
        best_value = np.inf
        best_point = None
        for phi in grid_phi:
            for tau in grid_tau:
                value = objective([np.log(phi), tau])
                if np.isfinite(value) and value < best_value and value < 1e11:
                    best_value = value
                    best_point = [np.log(phi), tau]
        if best_point is None:
            raise RuntimeError("no valid spatial covariance fit")
        phi = float(np.exp(best_point[0]))
        tau = float(best_point[1])
        fit_method = "grid_fallback"
    else:
        phi = float(np.exp(best.x[0]))
        tau = float(best.x[1])
        fit_method = "optimizer"

    bw, yw, _, factor = whitened(phi, tau)
    basis, rank, _ = orthonormal_basis(bw)
    resid = yw - basis @ (basis.T @ yw)
    sigma2 = float(np.sum(resid * resid) / max(n - rank, 1))
    if method == "vecchia":
        return Covariance(phi, tau, sigma2, None, True, fit_method=fit_method,
                          approximation="vecchia", factor=factor)
    chol = np.sqrt(sigma2) * factor
    return Covariance(phi, tau, sigma2, chol, True, fit_method=fit_method,
                      approximation="dense")

def profile(response, baseline, exposures, covariance, rcond=RCOND):

    # Whitened GLS compares every candidate exposure to the same spatially adjusted baseline.
    y = np.asarray(response, dtype=float)
    b = np.asarray(baseline, dtype=float)
    yw = covariance.whiten(y)
    bw = covariance.whiten(b)

    base_q, base_rank, _ = orthonormal_basis(bw, rcond)
    if base_rank == 0:
        raise ValueError("the whitened baseline has numerical rank zero")
    projected = base_q.T @ yw
    q0 = float(np.sum(yw * yw) - np.sum(projected * projected))
    baseline_coefficients = gls_fit(bw, yw, rcond)["coefficients"]

    n_points = len(exposures)
    values = np.empty(n_points)
    amplitudes = np.zeros(n_points)
    degenerate = np.zeros(n_points, dtype=bool)
    independence = np.zeros(n_points)
    bases = []

    threshold = rcond ** 0.5
    for k, h in enumerate(exposures):
        hw = covariance.whiten(np.asarray(h, dtype=float).ravel())
        residual = hw - base_q @ (base_q.T @ hw)
        norm_h = float(np.linalg.norm(hw))
        norm_r = float(np.linalg.norm(residual))
        ratio = norm_r / norm_h if norm_h > 0 else 0.0
        independence[k] = ratio

        if ratio <= threshold:
            degenerate[k] = True
            amplitudes[k] = 0.0
            values[k] = q0
            bases.append(base_q)
            continue

        direction = residual / norm_r
        projection = float(direction @ yw)
        amplitudes[k] = projection / norm_r
        values[k] = q0 - projection * projection
        bases.append(np.column_stack([base_q, direction]))

    return {
        "Q": values,
        "alpha": amplitudes,
        "Q0": q0,
        "degenerate": degenerate,
        "independence": independence,
        "baseline_rank": base_rank,
        "baseline_coefficients": baseline_coefficients,
        "whitened_baseline": bw,
        "whitened_response": yw,
        "orthonormal": bases,
        "baseline_orthonormal": base_q,
    }

def bootstrap_pvalue(fit, draws=BOOTSTRAP_DRAWS, seed=0):
    if draws <= 0:
        return None
    yw_mean = fit["whitened_baseline"] @ fit["baseline_coefficients"]
    n = yw_mean.size
    rng = np.random.default_rng(seed)
    observed = fit["Q0"] - float(np.min(fit["Q"]))
    exceed = 0
    base_q = fit["baseline_orthonormal"]
    for _ in range(draws):
        draw = yw_mean + rng.standard_normal(n)
        total = float(np.sum(draw * draw))
        q0 = total - float(np.sum((base_q.T @ draw) ** 2))
        best = np.inf
        # Repeat the full range search in every null draw.
        for q in fit["orthonormal"]:
            rss = total - float(np.sum((q.T @ draw) ** 2))
            if rss < best:
                best = rss
        if q0 - best >= observed:
            exceed += 1
    return float((1 + exceed) / (draws + 1))

def phi_bounds(median_nn_um, geodesic_diameter_um):
    return (
        max(10.0, 2.0 * float(median_nn_um)),
        min(1000.0, float(geodesic_diameter_um) / 3.0),
    )

def calibrated_secreted_pvalue(p_boot):
    # Wrong-ligand calibration is applied only to secreted-interaction bootstrap probabilities.
    p = np.asarray(p_boot, dtype=float)
    out = np.minimum(1.0, SECRETED_CALIBRATION_C * p ** SECRETED_CALIBRATION_GAMMA)
    return float(out) if out.ndim == 0 else out

def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out
