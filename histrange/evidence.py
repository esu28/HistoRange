import numpy as np

from .settings import SPECIFICITY_MIN_R2, MAX_RANGE_SPREAD_BITS, MAX_MESH_SHIFT_BITS

def checkerboard_folds(coordinates, blocks=8):
    # Checkerboard folds keep the two held-out sets spatially interleaved.
    coordinates = np.asarray(coordinates, dtype=float)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    xi = np.clip(((x - x.min()) / max(float(np.ptp(x)), 1e-9) * blocks).astype(int), 0, blocks - 1)
    yi = np.clip(((y - y.min()) / max(float(np.ptp(y)), 1e-9) * blocks).astype(int), 0, blocks - 1)
    return (xi + yi) % 2

def partial_r2(loss_without_target, loss_with_target):
    denominator = float(loss_without_target)
    if denominator <= 0:
        return np.nan
    return (denominator - float(loss_with_target)) / denominator

def specificity_supported(fold_r2, threshold=SPECIFICITY_MIN_R2):
    values = np.asarray(fold_r2, dtype=float)
    return bool(values.size >= 2 and np.all(values >= float(threshold)))

def geometry_gain(isotropic_loss, matched_loss, n_receivers):
    # Positive gain means matched H&E predicts held-out receivers better than isotropic transport.
    return (float(isotropic_loss) - float(matched_loss)) / max(int(n_receivers), 1)

def anisotropy_robust(ranges_um, max_spread_bits=MAX_RANGE_SPREAD_BITS):
    values = np.asarray(ranges_um, dtype=float)
    return bool(np.log2(values.max() / values.min()) <= float(max_spread_bits))

def response_robust(reference_um, perturbed_um, max_shift_bits=MAX_RANGE_SPREAD_BITS):
    values = np.asarray(perturbed_um, dtype=float)
    shift = np.max(np.abs(np.log2(values / float(reference_um))))
    return bool(shift <= float(max_shift_bits))

def mesh_robust(reference_um, adjacent_um, max_shift_bits=MAX_MESH_SHIFT_BITS):
    shift = abs(np.log2(float(adjacent_um) / float(reference_um)))
    return bool(shift <= float(max_shift_bits))
