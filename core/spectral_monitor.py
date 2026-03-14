"""
core/spectral_monitor.py — Spectral health diagnostics for square matrices.
============================================================================
Provides eigenvalue gap-ratio analysis, Frobenius distance from identity,
SVD decomposition, and a composite stability metric.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gmean


# GOE / GUE expected average gap ratios (Atas et al., 2013).
GOE_R_MEAN: float = 0.5307
GUE_R_MEAN: float = 0.5996
# Poisson (uncorrelated) baseline.
POISSON_R_MEAN: float = 0.3863
# Threshold below which we declare manifold collapse.
R_RATIO_FLOOR: float = 0.40


class StabilityBreak(Exception):
    """Raised when the gap ratio falls significantly below GOE/GUE averages,
    indicating loss of level repulsion (manifold collapse)."""


def get_r_ratio(evals: NDArray[np.floating]) -> float:
    """Average ratio of consecutive eigenvalue gaps (level-spacing statistic).

    For an ordered sequence of eigenvalues λ_1 ≤ λ_2 ≤ … ≤ λ_n, define
    gaps δ_i = λ_{i+1} − λ_i and the ratio

        r_i = min(δ_i, δ_{i+1}) / max(δ_i, δ_{i+1})

    The mean ⟨r⟩ distinguishes correlated spectra (GOE ≈ 0.53, GUE ≈ 0.60)
    from uncorrelated Poisson spectra (≈ 0.39).

    Parameters
    ----------
    evals : 1-D real array
        Eigenvalues (need not be sorted; will be sorted internally).

    Returns
    -------
    r_mean : float
        Average gap ratio in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 3 eigenvalues are provided (need at least 2 gaps).
    """
    evals = np.sort(np.asarray(evals, dtype=np.float64).ravel())
    if evals.size < 3:
        raise ValueError("Need at least 3 eigenvalues to compute gap ratios.")

    gaps = np.diff(evals)
    # Avoid division by zero for degenerate pairs.
    numerator = np.minimum(gaps[:-1], gaps[1:])
    denominator = np.maximum(gaps[:-1], gaps[1:])
    # Where both gaps are zero the ratio is defined as 0.
    with np.errstate(invalid="ignore"):
        ratios = np.where(denominator > 0, numerator / denominator, 0.0)

    return float(np.mean(ratios))


class TransportEvaluator:
    """Evaluate spectral transport properties of a square matrix.

    Parameters
    ----------
    matrix : 2-D array_like
        Square matrix to analyse.

    Attributes
    ----------
    matrix : ndarray
        The stored matrix (float64).
    n : int
        Dimension of the matrix.
    """

    def __init__(self, matrix: NDArray[np.floating]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Input must be a square 2-D matrix.")
        self.n: int = self.matrix.shape[0]

    # ------------------------------------------------------------------
    # Core decompositions
    # ------------------------------------------------------------------

    def frobenius_distance_from_identity(self) -> float:
        r"""Frobenius distance from the identity matrix.

        .. math:: \|A - I\|_F = \sqrt{\sum_{i,j} |a_{ij} - \delta_{ij}|^2}

        Returns
        -------
        float
        """
        diff = self.matrix - np.eye(self.n)
        return float(np.linalg.norm(diff, "fro"))

    def svd(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Full Singular Value Decomposition.

        Returns
        -------
        U : ndarray, shape (n, n)
        sigma : ndarray, shape (n,)   — singular values
        Vt : ndarray, shape (n, n)
        """
        return np.linalg.svd(self.matrix, full_matrices=True)

    # ------------------------------------------------------------------
    # Stability metric
    # ------------------------------------------------------------------

    def stability(self) -> float:
        r"""Composite stability metric.

        Combines two signals:
        1. **Singular-value variance** — how far the singular values deviate
           from uniformity (ideal unitary transport → all σ_i = 1).
        2. **Eigenvalue gap ratio** — level-repulsion health of the spectrum.

        .. math::
            S = \bigl(\operatorname{Var}(\sigma)\bigr)^{1/2}
                \;\cdot\; \langle r \rangle

        Low S (close to 0) indicates both tight singular-value clustering
        *and* healthy level repulsion.  A high variance or a collapsing
        gap ratio both inflate S, signalling instability.

        Returns
        -------
        float

        Raises
        ------
        StabilityBreak
            If the eigenvalue gap ratio drops below ``R_RATIO_FLOOR``.
        """
        _, sigma, _ = self.svd()
        sv_std = float(np.std(sigma))

        evals = np.linalg.eigvalsh(
            (self.matrix + self.matrix.T) / 2  # symmetrise for real spectrum
        )
        r_mean = get_r_ratio(evals)

        if r_mean < R_RATIO_FLOOR:
            raise StabilityBreak(
                f"Gap ratio ⟨r⟩ = {r_mean:.4f} < {R_RATIO_FLOOR} — "
                "manifold collapse detected."
            )

        return float(gmean([sv_std + 1e-12, r_mean]))
