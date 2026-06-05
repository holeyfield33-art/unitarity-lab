"""
core/bocpd.py — Two-model log-space Bayesian Online Changepoint Detection.
==========================================================================
Replaces the previous single-NIG-prior detector (which misused probability
density as probability and included a broken "surprise boost"). This
implementation is the validated LockedBOCPDMonitor design that cleanly
detects synthetic GUE→collapsed regime changes without false alarms on
stable streams.

Observation unit: r_ratio is fed directly (raw GUE ~0.64 stable, ~0.42
collapsed). The old fused x = (zeta**0.65)*(r_ratio**0.35) is NOT used
because all validated baselines and calibration data are in raw r-ratio
units. zeta is accepted by process_step for API compatibility but is not
used in the likelihood computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


class PredictiveAnomalyDetector:
    """Two-model log-space BOCPD for detecting GUE→collapse regime shifts.

    model_0 tracks the pre-change / GUE stable regime.
    model_1 tracks the post-change / collapsed regime.

    Parameters
    ----------
    hazard_rate : float
        Expected steps between changepoints; hazard = 1 / hazard_rate.
        Validated value is 1000 (hazard = 0.001).
    threshold : float
        Alarm threshold on P(changepoint) for external callers.
    max_run_length : int
        Maximum run-length hypotheses retained (older mass is pruned).
    mean_0 : float or None
        Mean of the stable-regime model. None triggers measured-baseline
        calibration from the first warmup_steps observations.
    std_0 : float
        Standard deviation of the stable-regime model.
    mean_1 : float
        Mean of the collapsed-regime model (default 0.42).
    std_1 : float
        Standard deviation of the collapsed-regime model.
    warmup_steps : int
        Number of initial observations to collect before running detection.
        During warm-up, process_step returns 0.0.
    """

    def __init__(
        self,
        hazard_rate: float = 1000.0,
        threshold: float = 0.95,
        max_run_length: int = 500,
        mean_0: Optional[float] = None,
        std_0: float = 0.015,
        mean_1: float = 0.42,
        std_1: float = 0.015,
        warmup_steps: int = 100,
    ) -> None:
        self.hazard = 1.0 / hazard_rate
        self.threshold = threshold
        self.max_run_length = max_run_length
        self._mean_0_init = mean_0           # None → calibrate from warm-up
        self.mean_0: float = mean_0 if mean_0 is not None else 0.0
        self.std_0 = std_0
        self.mean_1 = mean_1
        self.std_1 = std_1
        self.warmup_steps = warmup_steps

        # Pre-calibrated when mean_0 is provided explicitly
        self.calibrated: bool = mean_0 is not None
        self._warmup_done: bool = mean_0 is not None
        self._warmup_buffer: List[float] = []
        self.t: int = 0

        # Log-space run-length posterior; P(r_0 = 0) = 1 → log = 0
        self._log_R = np.array([0.0])

    # ------------------------------------------------------------------
    # Log-likelihood helpers (evaluated at call time; safe after calibration)
    # ------------------------------------------------------------------

    def _log_p0(self, x: float) -> float:
        return float(norm.logpdf(x, loc=self.mean_0, scale=self.std_0))

    def _log_p1(self, x: float) -> float:
        return float(norm.logpdf(x, loc=self.mean_1, scale=self.std_1))

    # ------------------------------------------------------------------
    # Warm-up calibration
    # ------------------------------------------------------------------

    def _finish_warmup(self) -> None:
        """Calibrate model_0 from the warm-up buffer (if mean_0 was None)."""
        buf = np.asarray(self._warmup_buffer)
        if self._mean_0_init is None:
            self.mean_0 = float(np.mean(buf))
            # Floor std to avoid an over-confident zero-variance model
            self.std_0 = float(max(float(np.std(buf)), 1e-3))
        self.calibrated = True
        self._warmup_done = True
        # Reset to a clean prior so warm-up observations don't pollute the posterior
        self._log_R = np.array([0.0])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_step(self, zeta: float, r_ratio: float) -> float:
        """Update the run-length posterior with one new observation.

        Observation is r_ratio (raw GUE spacing ratio, not fused with zeta).
        zeta is accepted for API compatibility with the orchestrator call site.

        Returns
        -------
        float
            P(changepoint at this step) in [0.0, 1.0].
            Returns 0.0 during the warm-up phase.
        """
        x = r_ratio   # raw r-ratio; GUE stable ~0.64, collapsed ~0.42
        self.t += 1

        # --- Warm-up phase: collect buffer, skip inference ---
        if not self._warmup_done:
            self._warmup_buffer.append(x)
            if len(self._warmup_buffer) >= self.warmup_steps:
                self._finish_warmup()
            return 0.0

        # --- Log-space two-model BOCPD recursion ---
        log_hazard = np.log(self.hazard)
        log_1m_hazard = np.log(1.0 - self.hazard)
        log_p0 = self._log_p0(x)
        log_p1 = self._log_p1(x)

        # Growth: run continued → P(r_t = k+1) ∝ P(r_{t-1}=k) · p0(x) · (1−h)
        log_growth = self._log_R + log_p0 + log_1m_hazard

        # Changepoint: run reset → P(r_t = 0) ∝ Σ_k P(r_{t-1}=k) · p1(x) · h
        # p1 likelihood on the first observation of the new (collapsed) regime.
        log_cp = float(logsumexp(self._log_R)) + log_p1 + log_hazard

        new_log_R = np.empty(len(self._log_R) + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize in log-space
        log_total = float(logsumexp(new_log_R))
        if np.isfinite(log_total):
            new_log_R -= log_total
        else:
            # Numerical fallback: uniform over current hypotheses
            new_log_R = np.full(len(new_log_R), -np.log(len(new_log_R)))

        # Prune old run-length hypotheses to cap memory
        if len(new_log_R) > self.max_run_length:
            new_log_R = new_log_R[: self.max_run_length]
            log_total = float(logsumexp(new_log_R))
            if np.isfinite(log_total):
                new_log_R -= log_total

        self._log_R = new_log_R
        return float(np.clip(np.exp(self._log_R[0]), 0.0, 1.0))

    def diagnostics(self) -> Dict[str, object]:
        """Current detector state for debugging and calibration audit."""
        return {
            "hazard": self.hazard,
            "threshold": self.threshold,
            "mean_0": self.mean_0,
            "std_0": self.std_0,
            "mean_1": self.mean_1,
            "std_1": self.std_1,
            "warmup_steps": self.warmup_steps,
            "calibrated": self.calibrated,
            "t": self.t,
        }


# Alias: the validated two-model detector is the canonical implementation
LockedBOCPDMonitor = PredictiveAnomalyDetector
