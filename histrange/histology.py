import numpy as np
from scipy import ndimage

from .settings import H_STAIN, E_STAIN, MASK_SIGMA_UM, OPENING_RADIUS_UM, CLOSING_RADIUS_UM, MIN_COMPONENT_AREA_UM2

class Histology:
    def __init__(self, mask, eosin, hematoxylin, od_norm, pixel_size_um):
        self.mask = mask
        self.eosin = eosin
        self.hematoxylin = hematoxylin
        self.od_norm = od_norm
        self.pixel_size_um = pixel_size_um

    @property
    def shape(self):
        return self.mask.shape

    def tissue_area_um2(self):
        return float(self.mask.sum()) * self.pixel_size_um ** 2

    def __repr__(self):
        return "Histology(%dx%d px, %.1f%% tissue)" % (
            self.mask.shape[0],
            self.mask.shape[1],
            100.0 * self.mask.mean(),
        )

def optical_density(image):
    img = np.asarray(image, dtype=float)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("histology image must be RGB with shape (rows, cols, 3)")
    img = img[:, :, :3]
    if img.max() <= 1.0 + 1e-9:
        img = img * 255.0
    # Optical density is the input to H&E color deconvolution.
    return -np.log((img + 1.0) / 256.0)

def stain_basis():
    h = H_STAIN / np.linalg.norm(H_STAIN)
    e = E_STAIN / np.linalg.norm(E_STAIN)
    r = np.cross(h, e)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        raise ValueError("stain vectors are collinear")
    return np.vstack([h, e, r / norm])

def separate_stains(od, basis):
    flat = od.reshape(-1, 3)
    # Solve for hematoxylin and eosin concentrations in the fixed stain basis.
    conc = np.linalg.solve(basis.T, flat.T).T
    conc = np.clip(conc, 0.0, None)
    return conc.reshape(od.shape[0], od.shape[1], 3)

def otsu_threshold(values, bins=256):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    lo = float(v.min())
    hi = float(v.max())
    if hi <= lo:
        return hi
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts).astype(float)
    total = weight[-1]
    moment = np.cumsum(counts * centers)
    grand = moment[-1]
    usable = (weight > 0) & (weight < total)
    mu_low = np.divide(moment, np.maximum(weight, 1.0))
    mu_high = np.divide(grand - moment, np.maximum(total - weight, 1.0))
    between = weight * (total - weight) * (mu_low - mu_high) ** 2
    between[~usable] = -1.0
    return float(centers[int(np.argmax(between))])

def disk(radius_px):
    if radius_px < 1.0:
        return None
    r = int(np.ceil(radius_px))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px + 1e-9

def tissue_mask(od, pixel_size_um, capture_mask=None):
    norm = np.linalg.norm(od, axis=2)
    sigma_px = MASK_SIGMA_UM / pixel_size_um
    smooth = ndimage.gaussian_filter(norm, sigma=max(sigma_px, 1e-6))

    values = smooth if capture_mask is None else smooth[np.asarray(capture_mask, dtype=bool)]
    threshold = otsu_threshold(values)
    mask = smooth > threshold
    if capture_mask is not None:
        mask &= np.asarray(capture_mask, dtype=bool)

    # Clean small artifacts without filling internal tissue holes.
    opening = disk(OPENING_RADIUS_UM / pixel_size_um)
    if opening is not None:
        mask = ndimage.binary_opening(mask, structure=opening)
    closing = disk(CLOSING_RADIUS_UM / pixel_size_um)
    if closing is not None:
        mask = ndimage.binary_closing(mask, structure=closing)

    min_area = MIN_COMPONENT_AREA_UM2
    min_pixels = int(np.ceil(min_area / pixel_size_um ** 2))
    if min_pixels > 1:
        labels, count = ndimage.label(mask)
        if count > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            keep = np.zeros(sizes.shape[0], dtype=bool)
            keep[sizes >= min_pixels] = True
            mask = keep[labels]

    if not mask.any():
        raise ValueError("tissue mask is empty; check the image or the mask settings")
    return mask

def prepare(image, pixel_size_um=1.0, capture_mask=None):
    od = optical_density(image)
    basis = stain_basis()
    conc = separate_stains(od, basis)
    mask = tissue_mask(od, pixel_size_um, capture_mask=capture_mask)
    return Histology(
        mask=mask,
        eosin=conc[:, :, 1],
        hematoxylin=conc[:, :, 0],
        od_norm=np.linalg.norm(od, axis=2),
        pixel_size_um=float(pixel_size_um),
    )
