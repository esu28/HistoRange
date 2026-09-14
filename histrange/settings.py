import numpy as np

# Fixed analysis settings reported in the manuscript and supplement.

H_STAIN = np.array([0.650, 0.704, 0.286], dtype=float)
E_STAIN = np.array([0.072, 0.990, 0.105], dtype=float)

MASK_SIGMA_UM = 2.0
OPENING_RADIUS_UM = 4.0
CLOSING_RADIUS_UM = 8.0
MIN_COMPONENT_AREA_UM2 = 500.0

DERIVATIVE_SCALE_UM = 2.0
TENSOR_SCALE_UM = 12.0
COHERENCY_FLOOR = 0.20
CROSSING_THRESHOLD = 0.35
BETA_MAX = 0.5 * np.log(3.0)

DEFAULT_MESH_UM = 4.0
FALLBACK_MESH_UM = (8.0, 16.0)
MAX_MESH_NODES = 2_000_000

RANGE_MIN_UM = 16.0
RANGE_MAX_UM = 512.0
RANGE_INTERVALS_PER_OCTAVE = 8
MIN_RANGE_POINTS = 9

NUGGET_BOUNDS = (0.05, 0.95)
NUGGET_START = 0.25
DENSE_GP_MAX_RECEIVERS = 2000
VECCHIA_NEIGHBORS = 30
FALLBACK_PHI_POINTS = 16
FALLBACK_NUGGET_GRID = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)
BOOTSTRAP_DRAWS = 9999

SOURCE_MIN_POSITIVE = 20
SOURCE_MIN_RAW_COUNTS = 25
RECEPTOR_MIN_COMPETENT = 30
STATE_MIN_FRACTION = 0.10
STATE_MIN_SIZE = 30

ANISOTROPY_MULTIPLIERS = (0.5, 1.0, 1.5)
SATURATION_K = (0.25, 0.5, 1.0, 2.0, 4.0)
MAX_RANGE_SPREAD_BITS = 1.0
MAX_MESH_SHIFT_BITS = 0.5
SPECIFICITY_MIN_R2 = 0.01

SECRETED_CALIBRATION_C = 0.6589
SECRETED_CALIBRATION_GAMMA = 0.7998

SHORT_RANGE_SPLIT_UM = 64.0
SHORT_BAND_BITS = {0.90: 1.312, 0.95: 1.750}
SHORT_UPPER_BITS = {0.90: 0.875, 0.95: 1.500}
LONG_UPPER_BITS = {0.90: 1.000, 0.95: 1.250}

def master_range_grid():
    n = int(round(np.log2(RANGE_MAX_UM / RANGE_MIN_UM) * RANGE_INTERVALS_PER_OCTAVE)) + 1
    return RANGE_MIN_UM * 2.0 ** (np.arange(n) / RANGE_INTERVALS_PER_OCTAVE)

def trusted_range_grid(mesh_spacing_um, median_nn_um, geodesic_diameter_um):
    grid = master_range_grid()
    low = max(RANGE_MIN_UM, 4.0 * float(mesh_spacing_um), 0.5 * float(median_nn_um))
    high = min(RANGE_MAX_UM, 0.25 * float(geodesic_diameter_um))
    keep = grid[(grid >= low - 1e-12) & (grid <= high + 1e-12)]
    if keep.size < MIN_RANGE_POINTS:
        raise ValueError("fewer than nine trusted range values remain")
    return keep
