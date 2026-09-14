# HistoRange

HistoRange is a computational framework for spatial transcriptomics that uses registered H&E histology to define a spatially varying anisotropic transport geometry for secreted ligand signaling. It separates spatial association detection from post-detection effective-range estimation so that a fitted scale is reported quantitatively only when the data support it.

The code implements the core HistoRange workflow, including molecular scoring, histology-derived transport geometry, finite-element transport, contact exposure, covariance-aware spatial association testing, range estimation, and the accompanying specificity, robustness, and histology-evidence calculations.
