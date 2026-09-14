import numpy as np
from scipy import ndimage

from .settings import DERIVATIVE_SCALE_UM, TENSOR_SCALE_UM, COHERENCY_FLOOR, CROSSING_THRESHOLD, BETA_MAX

PREFILTER = np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
DERIVATIVE = np.array([0.109604, 0.276691, 0.0, -0.276691, -0.109604])

EPS = 1e-12

class Geometry:
    def __init__(self, theta, coherency, crossing, beta, mask, pixel_size_um):
        self.theta = theta
        self.coherency = coherency
        self.crossing = crossing
        self.beta = beta
        self.mask = mask
        self.pixel_size_um = pixel_size_um

    def anisotropic_fraction(self):
        inside = self.mask
        if not inside.any():
            return 0.0
        return float((self.beta[inside] > 0).mean())

    def tensor_at(self, points_um, beta_scale=1.0):
        pts = np.atleast_2d(np.asarray(points_um, dtype=float))
        cols = np.clip(np.rint(pts[:, 0] / self.pixel_size_um).astype(int), 0, self.theta.shape[1] - 1)
        rows = np.clip(np.rint(pts[:, 1] / self.pixel_size_um).astype(int), 0, self.theta.shape[0] - 1)
        theta = self.theta[rows, cols]
        beta = self.beta[rows, cols] * float(beta_scale)
        inside = self.mask[rows, cols]
        beta = np.where(inside, beta, 0.0)
        return tensor_components(theta, beta)

    def __repr__(self):
        return "Geometry(%d x %d px, anisotropic %.1f%%)" % (
            self.theta.shape[0],
            self.theta.shape[1],
            100.0 * self.anisotropic_fraction(),
        )

def tensor_components(theta, beta):
    # exp(beta) and exp(-beta) keep det(A)=1, separating directionality from range.
    c = np.cos(theta)
    s = np.sin(theta)
    hi = np.exp(beta)
    lo = np.exp(-beta)
    a11 = c * c * hi + s * s * lo
    a22 = s * s * hi + c * c * lo
    a12 = c * s * (hi - lo)
    return np.stack([a11, a12, a22], axis=-1)

def _derivatives(channel):
    ex = ndimage.convolve1d(channel, DERIVATIVE, axis=1, mode="nearest")
    ex = ndimage.convolve1d(ex, PREFILTER, axis=0, mode="nearest")
    ey = ndimage.convolve1d(channel, DERIVATIVE, axis=0, mode="nearest")
    ey = ndimage.convolve1d(ey, PREFILTER, axis=1, mode="nearest")
    return ex, ey

def structure_tensor(channel, sigma_px):
    ex, ey = _derivatives(channel)
    jxx = ndimage.gaussian_filter(ex * ex, sigma_px)
    jxy = ndimage.gaussian_filter(ex * ey, sigma_px)
    jyy = ndimage.gaussian_filter(ey * ey, sigma_px)
    return ex, ey, jxx, jxy, jyy

def crossing_statistic(ex, ey, sigma_px):
    w = ex * ex + ey * ey
    phi = np.arctan2(ey, ex)
    denom = ndimage.gaussian_filter(w, sigma_px) + EPS
    # Fourth-order moments flag ambiguous crossing orientations.
    m2 = (
        ndimage.gaussian_filter(w * np.cos(2 * phi), sigma_px)
        + 1j * ndimage.gaussian_filter(w * np.sin(2 * phi), sigma_px)
    ) / denom
    m4 = (
        ndimage.gaussian_filter(w * np.cos(4 * phi), sigma_px)
        + 1j * ndimage.gaussian_filter(w * np.sin(4 * phi), sigma_px)
    ) / denom
    return np.abs(m4 - m2 ** 2)

def build(histology):
    px = histology.pixel_size_um
    sigma_g = DERIVATIVE_SCALE_UM / px
    sigma_t = TENSOR_SCALE_UM / px

    channel = ndimage.gaussian_filter(histology.eosin, max(sigma_g, 1e-6))
    ex, ey, jxx, jxy, jyy = structure_tensor(channel, max(sigma_t, 1e-6))

    trace = jxx + jyy
    spread = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coherency = spread / (trace + EPS)

    gradient_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    # The image-gradient direction is rotated by 90 degrees to follow tissue lines.
    theta = gradient_angle + 0.5 * np.pi

    crossing = crossing_statistic(ex, ey, max(sigma_t, 1e-6))

    floor = COHERENCY_FLOOR
    beta_max = BETA_MAX
    crossing_max = CROSSING_THRESHOLD

    # Weak or crossing orientations revert to isotropic transport.
    strength = np.clip((coherency - floor) / (1.0 - floor), 0.0, 1.0)
    beta = beta_max * strength * (crossing <= crossing_max)
    beta = np.where(histology.mask, beta, 0.0)

    return Geometry(
        theta=theta,
        coherency=coherency,
        crossing=crossing,
        beta=beta,
        mask=histology.mask,
        pixel_size_um=px,
    )

def isotropic(geometry):
    return Geometry(
        np.zeros_like(geometry.theta), geometry.coherency, geometry.crossing,
        np.zeros_like(geometry.beta), geometry.mask, geometry.pixel_size_um
    )

def rotated(geometry, degrees=90.0):
    return Geometry(
        geometry.theta + np.deg2rad(degrees), geometry.coherency, geometry.crossing,
        geometry.beta.copy(), geometry.mask, geometry.pixel_size_um
    )

def morphology_mismatch(geometry, block_um=64.0, seed=0):
    rng = np.random.default_rng(int(seed))
    block_px = max(1, int(round(block_um / geometry.pixel_size_um)))
    theta = geometry.theta.copy()
    beta = geometry.beta.copy()
    rows, cols = theta.shape
    nr, nc = rows // block_px, cols // block_px
    if nr < 2 or nc < 2:
        return Geometry(theta, geometry.coherency, geometry.crossing, beta, geometry.mask, geometry.pixel_size_um)
    blocks = [(i, j) for i in range(nr) for j in range(nc)]
    order = rng.permutation(len(blocks))
    new_theta = theta.copy()
    new_beta = beta.copy()
    for src_idx, dst_idx in enumerate(order):
        i0, j0 = blocks[src_idx]
        i1, j1 = blocks[dst_idx]
        src = (slice(i0 * block_px, (i0 + 1) * block_px), slice(j0 * block_px, (j0 + 1) * block_px))
        dst = (slice(i1 * block_px, (i1 + 1) * block_px), slice(j1 * block_px, (j1 + 1) * block_px))
        new_theta[dst] = theta[src]
        new_beta[dst] = beta[src]
    new_beta = np.where(geometry.mask, new_beta, 0.0)
    return Geometry(new_theta, geometry.coherency, geometry.crossing, new_beta, geometry.mask, geometry.pixel_size_um)
