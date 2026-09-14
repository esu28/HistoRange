import math
import numpy as np

from .settings import (
    RECEPTOR_MIN_COMPETENT,
    SOURCE_MIN_POSITIVE,
    SOURCE_MIN_RAW_COUNTS,
    STATE_MIN_FRACTION,
    STATE_MIN_SIZE,
)

def cp10k(raw, library_size):
    raw = np.asarray(raw, dtype=float)
    library_size = np.maximum(np.asarray(library_size, dtype=float), 1.0)
    return 1e4 * raw / library_size[..., None] if raw.ndim == 2 else 1e4 * raw / library_size

def complex_count(subunit_counts):
    x = np.asarray(subunit_counts, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("subunit_counts must have shape (observations, subunits)")
    # A required complex is limited by its least abundant subunit.
    return np.min(x, axis=1)

def bounded_score(raw_complex, library_size):
    # Map log-CP10K abundance to a bounded score using the specimen median positive level.
    z = np.log1p(cp10k(raw_complex, library_size))
    positive = z[z > 0]
    k = float(np.median(positive)) if positive.size else 1.0
    return z / (z + max(k, 1e-12))

def qc_mask(total_umi, detected_genes, mito_fraction=None):
    total_umi = np.asarray(total_umi)
    detected_genes = np.asarray(detected_genes)
    keep = (total_umi >= 500) & (detected_genes >= 200)
    if mito_fraction is not None:
        keep &= np.asarray(mito_fraction, dtype=float) <= 0.20
    return keep

def detected_gene_mask(counts):
    x = np.asarray(counts)
    if x.ndim != 2:
        raise ValueError("counts must have shape (observations, genes)")
    threshold = max(10, int(math.ceil(0.001 * x.shape[0])))
    return np.count_nonzero(x > 0, axis=0) >= threshold

def eligibility(ligand_subunits, receptor_subunits, library_size, states=None):
    ligand_subunits = np.asarray(ligand_subunits, dtype=float)
    receptor_subunits = np.asarray(receptor_subunits, dtype=float)
    if ligand_subunits.ndim == 1:
        ligand_subunits = ligand_subunits[:, None]
    if receptor_subunits.ndim == 1:
        receptor_subunits = receptor_subunits[:, None]
    if ligand_subunits.shape[0] != receptor_subunits.shape[0]:
        raise ValueError("ligand and receptor arrays must have the same number of observations")

    library_size = np.asarray(library_size, dtype=float)
    # Every required subunit must pass the raw-count and CP10K thresholds.
    ligand_positive = np.all((ligand_subunits >= 1) & (cp10k(ligand_subunits, library_size) >= 1), axis=1)
    receptor_competent = np.all((receptor_subunits >= 1) & (cp10k(receptor_subunits, library_size) >= 1), axis=1)

    n_source = int(ligand_positive.sum())
    total_source = int(ligand_subunits.sum())
    n_receptor = int(receptor_competent.sum())
    state_fraction = None
    state_ok = True

    if states is not None:
        states = np.asarray(states)
        if states.shape[0] != receptor_competent.shape[0]:
            raise ValueError("states must match the number of observations")
        fractions = []
        for state in np.unique(states):
            mask = states == state
            if mask.sum() >= STATE_MIN_SIZE:
                fractions.append(float(receptor_competent[mask].mean()))
        state_fraction = max(fractions) if fractions else 0.0
        state_ok = state_fraction >= STATE_MIN_FRACTION

    return {
        "eligible": bool(
            n_source >= SOURCE_MIN_POSITIVE
            and total_source >= SOURCE_MIN_RAW_COUNTS
            and n_receptor >= RECEPTOR_MIN_COMPETENT
            and state_ok
        ),
        "n_positive_source": n_source,
        "total_raw_source": total_source,
        "n_competent_receivers": n_receptor,
        "max_state_fraction": state_fraction,
        "ligand_positive": ligand_positive,
        "receptor_competent": receptor_competent,
    }
