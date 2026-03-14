"""
core/orchestrator.py — Spectral coherence loop and session auditing.
====================================================================
Integrates :class:`TransportEvaluator` (spectral_monitor) and
:class:`ResonanceStore` (resonance_kernel) into a single ingestion
pipeline that validates each hidden state before committing it to the
spectral memory.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .resonance_kernel import ResonanceStore
from .spectral_monitor import TransportEvaluator, get_r_ratio
from .validator import (
    ModelStats,
    evaluate_model_health,
    log_audit,
    parse_metrics_from_text,
)

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────
COHERENCE_R_WARN: float = 0.45
"""Gap-ratio threshold below which a SymmetryBreakWarning is issued."""

MIN_EIGENVALUES: int = 4
"""Minimum accumulated vectors before gap-ratio checks are meaningful."""


# ── custom warning ────────────────────────────────────────────────────
class SymmetryBreakWarning(UserWarning):
    """Issued when ⟨r⟩ drops below the coherence threshold."""


# ── per-step record ──────────────────────────────────────────────────
@dataclass
class StepRecord:
    """Diagnostic snapshot captured for every ingested state."""

    step: int
    r_ratio: Optional[float]
    spectral_density: float
    frobenius_dist: float
    warned: bool


class Orchestrator:
    """Spectral coherence loop.

    Parameters
    ----------
    dim : int
        Dimensionality of the hidden-state vectors.
    r_warn : float
        Gap-ratio threshold for the SymmetryBreakWarning (default 0.45).
    eta : float
        Green's function regulariser forwarded to :class:`ResonanceStore`.
    top_k : int or None
        Principal directions for resonance scoring.
    """

    def __init__(
        self,
        dim: int,
        *,
        r_warn: float = COHERENCE_R_WARN,
        eta: float = 1e-3,
        top_k: Optional[int] = None,
    ) -> None:
        self.dim = dim
        self.r_warn = r_warn
        self._store = ResonanceStore(dim, eta=eta, top_k=top_k)
        self._history: List[StepRecord] = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def ingest(self, hidden_state: NDArray[np.floating]) -> StepRecord:
        """Process a single hidden-state vector.

        1. Build the outer-product matrix vvᵀ and wrap it in a
           :class:`TransportEvaluator` to get a Frobenius distance from I.
        2. If enough vectors have accumulated, compute the gap ratio ⟨r⟩
           on the running kernel's spectrum.  If ⟨r⟩ < ``r_warn``, issue
           a :class:`SymmetryBreakWarning` **before** storing.
        3. Commit the vector to the :class:`ResonanceStore` and record
           the spectral density at the new state.

        Parameters
        ----------
        hidden_state : 1-D array of length ``dim``.

        Returns
        -------
        StepRecord
        """
        v = np.asarray(hidden_state, dtype=np.float64).ravel()
        if v.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of length {self.dim}, got {v.shape[0]}."
            )

        # -- transport diagnostics on the rank-1 outer product -----------
        outer = np.outer(v, v)
        te = TransportEvaluator(outer)
        frob = te.frobenius_distance_from_identity()

        # -- gap-ratio coherence check on the cumulative kernel ----------
        warned = False
        r_ratio: Optional[float] = None

        # The kernel already contains contributions from previous steps;
        # peek at what the spectrum will look like *after* adding v.
        candidate_kernel = self._store._kernel + outer
        n_accumulated = self._store.size + 1

        if n_accumulated >= MIN_EIGENVALUES:
            sym = (candidate_kernel + candidate_kernel.T) / 2
            evals = np.linalg.eigvalsh(sym)
            r_ratio = get_r_ratio(evals)

            if r_ratio < self.r_warn:
                msg = (
                    f"Step {self._step}: ⟨r⟩ = {r_ratio:.4f} < {self.r_warn} "
                    "— symmetry degradation detected."
                )
                warnings.warn(msg, SymmetryBreakWarning, stacklevel=2)
                logger.warning(msg)
                warned = True

        # -- commit to spectral memory -----------------------------------
        self._store.store(v, label=self._step)
        density = self._store.spectral_density(v)

        record = StepRecord(
            step=self._step,
            r_ratio=r_ratio,
            spectral_density=density,
            frobenius_dist=frob,
            warned=warned,
        )
        self._history.append(record)
        self._step += 1
        return record

    # ------------------------------------------------------------------
    # Retrieval pass-through
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: NDArray[np.floating],
        top_n: int = 5,
    ):
        """Delegate to the underlying :class:`ResonanceStore`."""
        return self._store.retrieve(query, top_n=top_n)

    # ------------------------------------------------------------------
    # Audit / visualisation
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[StepRecord]:
        """Full step-by-step diagnostic history."""
        return list(self._history)

    def evaluate_health(self) -> "HealthReport":  # noqa: F821
        """Compare the current session averages against the benchmark.

        Computes mean ⟨r⟩ and mean spectral density over all steps,
        uses the last Frobenius distance, and delegates to
        :func:`~core.validator.evaluate_model_health`.

        Returns
        -------
        HealthReport
        """
        from .validator import HealthReport  # local to avoid circular at module level

        if not self._history:
            return HealthReport(details="No steps recorded.")

        valid_r = [r.r_ratio for r in self._history if r.r_ratio is not None]
        mean_r = sum(valid_r) / len(valid_r) if valid_r else None
        mean_density = sum(r.spectral_density for r in self._history) / len(self._history)
        last_frob = self._history[-1].frobenius_dist

        stats = ModelStats(
            r_ratio=mean_r,
            zeta=mean_density,
            frobenius_stability=last_frob,
        )
        return evaluate_model_health(stats)

    def log_session_audit(self, *, tag: str = "") -> "Path":
        """Evaluate health and write a JSON audit log.

        Returns
        -------
        Path
            Location of the written JSON file.
        """
        report = self.evaluate_health()

        valid_r = [r.r_ratio for r in self._history if r.r_ratio is not None]
        stats = ModelStats(
            r_ratio=(sum(valid_r) / len(valid_r)) if valid_r else None,
            zeta=(
                sum(r.spectral_density for r in self._history) / len(self._history)
                if self._history
                else None
            ),
            frobenius_stability=(
                self._history[-1].frobenius_dist if self._history else None
            ),
        )
        return log_audit(report, stats, tag=tag)

    def audit_session(self, save_path: Optional[str] = None) -> None:
        """Plot spectral density and gap ratio over the session.

        Parameters
        ----------
        save_path : str or None
            If given, save the figure to this path instead of showing it.
            Supports any format that matplotlib accepts (.png, .pdf, …).
        """
        # Import matplotlib lazily so the module works headless.
        import matplotlib

        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._history:
            logger.info("No steps recorded yet — nothing to audit.")
            return

        steps = [r.step for r in self._history]
        densities = [r.spectral_density for r in self._history]
        r_ratios = [r.r_ratio for r in self._history]
        warn_steps = [r.step for r in self._history if r.warned]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Spectral density (left axis).
        color_d = "#1f77b4"
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Spectral Density ρ(v)", color=color_d)
        ax1.plot(steps, densities, color=color_d, linewidth=1.2, label="ρ(v)")
        ax1.tick_params(axis="y", labelcolor=color_d)

        # Gap ratio (right axis).
        ax2 = ax1.twinx()
        color_r = "#ff7f0e"
        ax2.set_ylabel("Gap Ratio ⟨r⟩", color=color_r)
        valid = [(s, r) for s, r in zip(steps, r_ratios) if r is not None]
        if valid:
            s_vals, r_vals = zip(*valid)
            ax2.plot(s_vals, r_vals, color=color_r, linewidth=1.2, label="⟨r⟩")
            ax2.axhline(
                self.r_warn, color=color_r, linestyle="--", alpha=0.5,
                label=f"warn = {self.r_warn}",
            )
        ax2.tick_params(axis="y", labelcolor=color_r)

        # Mark warning steps.
        if warn_steps:
            for ws in warn_steps:
                ax1.axvline(ws, color="red", alpha=0.25, linewidth=0.8)

        fig.suptitle("Session Audit — Spectral Density & Gap Ratio")
        fig.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("Audit figure saved to %s", path)
        else:
            plt.show()
