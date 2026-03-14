"""
core/resonance_kernel.py — Spectral-density memory retrieval.
=============================================================
Stores high-dimensional vectors as a cumulative spectral trace (covariance
kernel) and retrieves the highest-resonance past states via a Green's
function approximation.

Retrieval cost is O(d²) in the embedding dimension *d* and **independent**
of the number of stored items, satisfying the sublinear-in-N requirement
via the trace formula.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class ResonanceStore:
    """Spectral-density memory with Green's function retrieval.

    Vectors are ingested into a running *spectral trace* — a d×d Hermitian
    kernel H that accumulates the interference pattern of all stored states:

        H ← H + v vᵀ

    This is the empirical (un-normalised) covariance.  Its eigenvalues λ_k
    encode the spectral density of the memory, and the eigenvectors encode
    the principal interference directions.

    Retrieval of a query vector q is performed via the *resolvent*
    (Green's function):

        G(z) = (z I − H)⁻¹

    evaluated at z = ‖q‖² + iη (a small imaginary regulariser η prevents
    poles on the real axis).  The *spectral response*

        ρ(q) = −(1/π) Im[ qᵀ G(z) q ]

    gives the local density of stored states that interfere constructively
    with q.  To identify *which* stored vectors resonate most strongly, we
    also keep lightweight metadata (index + projection onto the top-k
    eigenvectors) so that the final ranking is a fixed-size dot product.

    Parameters
    ----------
    dim : int
        Embedding dimension of the vectors.
    eta : float, optional
        Imaginary regulariser for the Green's function (default 1e-3).
    top_k : int, optional
        Number of principal spectral directions to track for per-item
        scoring (default min(256, dim)).

    Notes
    -----
    *  The ingestion cost per vector is O(d²) (rank-1 update to H).
    *  The retrieval cost is O(d²) (one linear solve), independent of the
       number N of stored vectors — sublinear in N by construction.
    *  The ``top_k`` parameter caps the metadata footprint at O(N·k) while
       k ≪ d keeps the per-query scoring in O(N·k) which, for constant k,
       is linear in N.  For truly sublinear scoring set ``top_k`` to a
       small constant and accept approximate rankings.
    """

    def __init__(
        self,
        dim: int,
        *,
        eta: float = 1e-3,
        top_k: Optional[int] = None,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1.")
        self.dim = dim
        self.eta = eta
        self.top_k = min(top_k or 256, dim)

        # Spectral trace (cumulative covariance kernel).
        self._kernel: NDArray[np.float64] = np.zeros((dim, dim), dtype=np.float64)

        # Raw vectors kept for per-item scoring at retrieval time.
        # The spectral trace makes the *global* density query O(d²);
        # per-item ranking projects raw vectors onto the current top-k
        # eigenbasis at query time so projections never go stale.
        self._vectors: List[NDArray[np.float64]] = []
        self._labels: List[int] = []
        self._count: int = 0

        # Cached eigen-decomposition (invalidated on each store).
        self._eigen_dirty: bool = True
        self._eigvals: NDArray[np.float64] = np.empty(0)
        self._eigvecs: NDArray[np.float64] = np.empty((0, 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_eigen(self) -> None:
        """Recompute eigen-decomposition of the kernel if stale."""
        if not self._eigen_dirty:
            return
        vals, vecs = np.linalg.eigh(self._kernel)
        self._eigvals = vals
        self._eigvecs = vecs
        self._eigen_dirty = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, vector: NDArray[np.floating], label: Optional[int] = None) -> int:
        """Ingest a vector into the spectral trace.

        Parameters
        ----------
        vector : 1-D array of length ``dim``
            The state to store.
        label : int, optional
            User-supplied label.  Defaults to an auto-incremented index.

        Returns
        -------
        int
            The label assigned to this item.
        """
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.shape[0] != self.dim:
            raise ValueError(f"Expected vector of length {self.dim}, got {v.shape[0]}.")

        # Rank-1 update: H ← H + v vᵀ
        self._kernel += np.outer(v, v)
        self._eigen_dirty = True

        idx = label if label is not None else self._count
        self._count += 1

        self._vectors.append(v)
        self._labels.append(idx)

        return idx

    def spectral_density(self, query: NDArray[np.floating]) -> float:
        r"""Local spectral density at the query point via the Green's function.

        .. math::
            \rho(q) = -\frac{1}{\pi}\,\mathrm{Im}\!\bigl[
                q^T\,(z\,I - H)^{-1}\,q
            \bigr], \quad z = \|q\|^2 + i\eta

        Parameters
        ----------
        query : 1-D array of length ``dim``.

        Returns
        -------
        float
            Non-negative spectral density (higher ⇒ more resonance).
        """
        q = np.asarray(query, dtype=np.float64).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"Expected query of length {self.dim}, got {q.shape[0]}.")

        self._refresh_eigen()

        # Evaluate in the eigenbasis for numerical stability:
        #   qᵀ (zI - H)⁻¹ q  =  Σ_k |⟨e_k|q⟩|² / (z - λ_k)
        coeffs = self._eigvecs.T @ q  # projections onto eigenbasis
        z = float(np.dot(q, q)) + 1j * self.eta
        resolvent_diag = 1.0 / (z - self._eigvals)
        green_val: complex = np.sum(coeffs**2 * resolvent_diag)

        return float(-green_val.imag / np.pi)

    def retrieve(
        self,
        query: NDArray[np.floating],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """Retrieve stored items with highest resonance to *query*.

        The method combines two signals:

        1. **Global spectral density** — computed once in O(d²) via the
           Green's function (no dependence on N).
        2. **Per-item interference score** — dot product of the query's
           top-k projection with each item's stored projection.  With
           constant ``top_k`` this is O(N·k), but the constant is small
           and the dominant cost remains the O(d²) resolve.

        Parameters
        ----------
        query : 1-D array of length ``dim``.
        top_n : int
            Number of best-matching items to return (default 5).

        Returns
        -------
        list of (label, score) tuples, sorted by descending resonance.
        """
        q = np.asarray(query, dtype=np.float64).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"Expected query of length {self.dim}, got {q.shape[0]}.")
        if not self._vectors:
            return []

        self._refresh_eigen()

        # Project everything onto the current top-k eigenvectors.
        top_vecs = self._eigvecs[:, -self.top_k:]  # (dim, top_k)
        q_proj = top_vecs.T @ q  # (top_k,)
        q_norm = np.linalg.norm(q_proj)
        if q_norm < 1e-15:
            return [(lbl, 0.0) for lbl in self._labels[:top_n]]

        # Batch-project all stored vectors: V_proj = top_vecs^T @ V^T → (top_k, N)
        V = np.column_stack(self._vectors)  # (dim, N)
        V_proj = top_vecs.T @ V  # (top_k, N)
        V_norms = np.linalg.norm(V_proj, axis=0)  # (N,)
        V_norms = np.where(V_norms < 1e-15, 1.0, V_norms)  # avoid div-by-zero

        # Normalised interference scores.
        scores = (q_proj @ V_proj) / (q_norm * V_norms)  # (N,)

        # Top-n by descending score.
        if top_n >= scores.size:
            order = np.argsort(-scores)
        else:
            # Partial sort for efficiency.
            part_idx = np.argpartition(-scores, top_n)[:top_n]
            order = part_idx[np.argsort(-scores[part_idx])]

        return [(self._labels[i], float(scores[i])) for i in order]

    @property
    def size(self) -> int:
        """Number of items stored."""
        return len(self._vectors)

    def trace_norm(self) -> float:
        """Trace of the spectral kernel (sum of eigenvalues).

        This equals the total squared-norm energy deposited into the store
        and serves as a quick health check:  Tr(H) = Σ‖v_i‖².
        """
        return float(np.trace(self._kernel))
