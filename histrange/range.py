import numpy as np
from scipy.special import gammaln

from .settings import SHORT_RANGE_SPLIT_UM, SHORT_BAND_BITS, SHORT_UPPER_BITS, LONG_UPPER_BITS

def _nb_irls(y, design, offset, dispersion, max_iter=25, ridge=1e-8):
    y = np.asarray(y, float)
    design = np.asarray(design, float)
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6)) - offset.mean()

    for _ in range(max_iter):
        eta = np.clip(offset + design @ beta, -25, 25)
        mu = np.exp(eta)
        weight = mu / (1.0 + dispersion * mu)
        working = eta - offset + (y - mu) / np.maximum(mu, 1e-12)
        weighted = design * weight[:, None]
        hessian = design.T @ weighted + ridge * np.eye(design.shape[1])
        updated = np.linalg.solve(hessian, weighted.T @ working)
        if not np.all(np.isfinite(updated)):
            break
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated

    eta = np.clip(offset + design @ beta, -25, 25)
    mu = np.exp(eta)
    if dispersion <= 1e-10:
        loglik = float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu))
    else:
        size = 1.0 / dispersion
        loglik = float(
            np.sum(
                gammaln(y + size)
                - gammaln(size)
                - gammaln(y + 1)
                + size * np.log(size / (size + mu))
                + y * np.log(mu / (size + mu))
            )
        )
    return beta, mu, loglik

def _moment_dispersion(y, mu):
    residual = (y - mu) ** 2 - mu
    value = residual.sum() / max((mu**2).sum(), 1e-12)
    return float(np.clip(value, 0.0, 25.0))

def fit_range_profile(counts, library_size, covariates, exposures, ranges):

    # T1 uses all bins, including receptor-zero bins, for post-detection range estimation.
    y = np.asarray(counts, float)
    offset = np.log(np.maximum(np.asarray(library_size, float), 1.0))
    covariates = np.asarray(covariates, float)
    baseline = np.column_stack([np.ones(y.size), covariates])

    _, mu0, _ = _nb_irls(y, baseline, offset, 0.0)
    # Estimate dispersion under the no-exposure model and hold it fixed across ranges.
    dispersion = _moment_dispersion(y, mu0)
    _, _, null_loglik = _nb_irls(y, baseline, offset, dispersion)

    loglik = []
    amplitudes = []
    for exposure in exposures:
        exposure = np.asarray(exposure, float)
        exposure = (exposure - exposure.mean()) / max(exposure.std(), 1e-12)
        beta, _, value = _nb_irls(
            y,
            np.column_stack([baseline, exposure]),
            offset,
            dispersion,
        )
        loglik.append(value)
        amplitudes.append(float(beta[-1]))

    # The range-support profile is twice the log-likelihood gain over the no-exposure model.
    profile = 2.0 * (np.asarray(loglik) - null_loglik)
    best_index = int(np.argmax(profile))
    return {
        "Lambda": float(profile[best_index]),
        "ell_hat": float(np.asarray(ranges, float)[best_index]),
        "best_index": best_index,
        "profile": profile,
        "dispersion": dispersion,
        "amplitude": amplitudes[best_index],
        "n": int(y.size),
        "loglik_null": null_loglik,
    }

def report_range(ell_hat, best_index, n_ranges, lambda_t1, support_threshold, grid_min, grid_max, level=0.95):
    ell_hat = float(ell_hat)
    if best_index <= 0 or best_index >= int(n_ranges) - 1:
        return {"status": "unresolved", "reason": "profile maximum is on the trusted-grid boundary"}
    if not np.isfinite(support_threshold) or float(lambda_t1) <= float(support_threshold):
        return {"status": "unresolved", "reason": "range-support threshold not passed"}
    if level not in (0.90, 0.95):
        raise ValueError("level must be 0.90 or 0.95")

    grid_min = float(grid_min)
    grid_max = float(grid_max)
    if ell_hat < SHORT_RANGE_SPLIT_UM:
        width = SHORT_BAND_BITS[level]
        low = ell_hat / 2.0 ** width
        high = ell_hat * 2.0 ** width
        if low > grid_min and high <= grid_max:
            return {"status": "interval", "lower_um": low, "upper_um": high, "level": level}
        # If the lower edge hits the trusted floor, use the calibrated upper bound.
        upper = ell_hat * 2.0 ** SHORT_UPPER_BITS[level]
    else:
        upper = ell_hat * 2.0 ** LONG_UPPER_BITS[level]

    if upper > grid_max:
        return {"status": "uninformative", "level": level}
    return {"status": "upper_bound", "upper_um": upper, "level": level}

def support_threshold(near_null_statistics):
    values = np.asarray(near_null_statistics, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 0.99))
