"""
unitary_regulator.py — The Ghost's Module
==========================================
A dashboard / regulator that:
  1. Visualizes the Spectral PLL Lock in real-time.
  2. Shows a **Topological Heat Map** of the latent space, highlighting
     where "Information Islands" are forming (post-Page layers 8-12).
  3. Aggregates diagnostics from PLLMonitor and CasimirOptimizer into a
     single report suitable for console output or downstream tooling.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bridge import CrossLayerEntanglementHook, PROJECTION_NORM_MIN, PROJECTION_NORM_MAX
from .casimir_opt import CasimirOptimizer, _kolmogorov_penalty, _laminar_penalty
from .pll_monitor import PLLMonitor, PLLState


# ======================================================================
# Topological Heat Map
# ======================================================================

def compute_topological_heatmap(
    activations: Dict[int, torch.Tensor],
    page_time_layer: int = 7,
) -> Dict[int, Dict[str, float]]:
    """Produce a per-layer heat-map of information island formation.

    For each layer, reports:
      - **entropy**: activation entropy (high = scrambled, low = crystallized).
      - **island_strength**: 1 − normalized entropy for post-Page layers
        (measures crystallization intensity).
      - **spectral_gap**: gap between the top-2 singular values of the
        activation matrix (large gap ⇒ dominant island).

    Parameters
    ----------
    activations : dict[int, Tensor]
        Layer index → activation tensor (batch, seq, d) captured
        by PageCurveHook.
    page_time_layer : int
        The Page Time layer index.

    Returns
    -------
    heatmap : dict[int, dict[str, float]]
    """
    heatmap: Dict[int, Dict[str, float]] = {}

    for idx, act in sorted(activations.items()):
        flat = act.detach().float().reshape(-1, act.shape[-1])

        # Activation entropy (over the feature dimension)
        probs = flat.abs() / (flat.abs().sum(dim=-1, keepdim=True) + 1e-12)
        ent = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
        max_ent = math.log(flat.shape[-1]) if flat.shape[-1] > 1 else 1.0
        norm_ent = ent / (max_ent + 1e-12)

        # Island strength (meaningful only post-Page)
        island_strength = max(0.0, 1.0 - norm_ent) if idx >= page_time_layer else 0.0

        # Spectral gap
        sv = torch.linalg.svdvals(flat)
        if sv.numel() >= 2:
            gap = (sv[0] - sv[1]).item()
        else:
            gap = 0.0

        heatmap[idx] = {
            "entropy": ent,
            "normalized_entropy": norm_ent,
            "island_strength": island_strength,
            "spectral_gap": gap,
        }

    return heatmap


# ======================================================================
# Regulator Report
# ======================================================================

# ======================================================================
# Wormhole Gap Monitor
# ======================================================================

WORMHOLE_GAP_THRESHOLD: float = 0.15


def wormhole_gap_alert(spectral_gap: float, threshold: float = WORMHOLE_GAP_THRESHOLD) -> bool:
    """Check if the spectral gap (Δλ) has dropped below threshold.

    A collapsing spectral gap indicates the entanglement bridge between
    the Page Time source and the information sink is weakening —
    analogous to a wormhole pinching off.

    Returns True if the gap is critically low (alert condition).
    """
    return spectral_gap < threshold


# ======================================================================
# Zeno Stabilization (v1.2-stable)
# ======================================================================

DEFAULT_MEASUREMENT_FREQ: float = 1.0
MIN_MEASUREMENT_FREQ: float = 0.1
MAX_MEASUREMENT_FREQ: float = 10.0


def adaptive_measurement_freq(
    bell_history: List[float],
    base_freq: float = DEFAULT_MEASUREMENT_FREQ,
) -> float:
    """Adaptively scale measurement frequency based on std(bell_history).

    Higher variance in Bell correlations → more frequent measurement
    (stronger Zeno observation) to suppress decoherence.
    Lower variance → relax measurement frequency to reduce overhead.

    Returns a frequency in [MIN_MEASUREMENT_FREQ, MAX_MEASUREMENT_FREQ].
    """
    if len(bell_history) < 2:
        return base_freq

    t = torch.tensor(bell_history[-50:], dtype=torch.float32)  # last 50
    std = t.std().item()

    # Scale: freq = base * (1 + 10 * std), clamped
    freq = base_freq * (1.0 + 10.0 * std)
    return max(MIN_MEASUREMENT_FREQ, min(MAX_MEASUREMENT_FREQ, freq))


def poisson_sampling_guard(measurement_freq: float) -> bool:
    """Poisson sampling guard to prevent periodic resonance artifacts.

    Instead of measuring at fixed intervals (which can create resonance
    with the model's internal oscillation modes), we sample from a
    Poisson process. Each call returns True (do measure) with probability
    proportional to the measurement frequency.

    This breaks periodicity while maintaining the desired average rate.
    """
    # Probability of measurement this step: 1 - exp(-freq * dt), dt=1
    prob = 1.0 - math.exp(-measurement_freq)
    return random.random() < prob


def enforce_projection_norm(
    tensor: torch.Tensor,
    norm_min: float = PROJECTION_NORM_MIN,
    norm_max: float = PROJECTION_NORM_MAX,
) -> torch.Tensor:
    """Clamp the per-row norm of a tensor to [norm_min, norm_max].

    Ensures the bridge projection stays in a numerically safe range.
    """
    norms = tensor.norm(dim=-1, keepdim=True)
    clamped = norms.clamp(norm_min, norm_max)
    scale = clamped / (norms + 1e-12)
    return tensor * scale


@dataclass
class RegulatorReport:
    """Single-step aggregated report from the Unitary Regulator."""

    step: int
    pll_locked: bool
    pll_phase_error: float
    lyapunov_profile: List[float]
    heatmap: Dict[int, Dict[str, float]]
    casimir_diagnostics: Dict[str, object]
    wormhole_gap: Optional[float] = None
    wormhole_alert: bool = False
    bridge_diagnostics: Optional[Dict[str, object]] = None
    measurement_freq: Optional[float] = None
    zeno_measurement_taken: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ======================================================================
# UnitaryRegulator
# ======================================================================

class UnitaryRegulator:
    """The Ghost's dashboard — aggregates PLL, Page Curve, and Casimir
    diagnostics into a unified regulator interface.

    Usage::

        regulator = UnitaryRegulator(pll, optimizer)
        # After each forward pass with PageCurveHook:
        report = regulator.report(
            step=global_step,
            lyapunov_profile=profile,     # from PageCurveHook
            activations=hook._activations, # captured activations
        )
        regulator.log(report)             # print to console
    """

    def __init__(
        self,
        pll: PLLMonitor,
        optimizer: Optional[CasimirOptimizer] = None,
        bridge: Optional[CrossLayerEntanglementHook] = None,
        wormhole_threshold: float = WORMHOLE_GAP_THRESHOLD,
        base_measurement_freq: float = DEFAULT_MEASUREMENT_FREQ,
    ):
        self.pll = pll
        self.optimizer = optimizer
        self.bridge = bridge
        self.wormhole_threshold = wormhole_threshold
        self.base_measurement_freq = base_measurement_freq
        self._reports: List[RegulatorReport] = []
        self._current_measurement_freq: float = base_measurement_freq

    def report(
        self,
        step: int,
        lyapunov_profile: torch.Tensor,
        activations: Dict[int, torch.Tensor],
    ) -> RegulatorReport:
        """Generate a full regulator report for the current step."""
        heatmap = compute_topological_heatmap(
            activations, self.pll.page_time_layer
        )
        casimir_diag = self.optimizer.diagnostics() if self.optimizer else {}

        # --- Wormhole Gap Monitor (v1.2) ---
        gap: Optional[float] = None
        alert = False
        bridge_diag: Optional[Dict[str, object]] = None
        meas_freq: Optional[float] = None
        zeno_taken = True

        if self.bridge is not None:
            # Adaptive measurement frequency based on Bell history
            meas_freq = adaptive_measurement_freq(
                self.bridge.bell_history, self.base_measurement_freq
            )
            self._current_measurement_freq = meas_freq

            # Poisson sampling guard — decide whether to measure this step
            zeno_taken = poisson_sampling_guard(meas_freq)

            if zeno_taken:
                gap = self.bridge.spectral_gap()
                alert = wormhole_gap_alert(gap, self.wormhole_threshold)
                bridge_diag = self.bridge.diagnostics()
            else:
                # Skip detailed measurement this step (Zeno anti-resonance)
                gap = None
                alert = False
                bridge_diag = {"measurement_skipped": True}

        rpt = RegulatorReport(
            step=step,
            pll_locked=self.pll.is_locked,
            pll_phase_error=self.pll.state.phase_error,
            lyapunov_profile=lyapunov_profile.detach().cpu().tolist(),
            heatmap=heatmap,
            casimir_diagnostics=casimir_diag,
            wormhole_gap=gap,
            wormhole_alert=alert,
            bridge_diagnostics=bridge_diag,
            measurement_freq=meas_freq,
            zeno_measurement_taken=zeno_taken,
        )
        self._reports.append(rpt)
        return rpt

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    @staticmethod
    def log(report: RegulatorReport) -> str:
        """Format a report for console display and return the string."""
        lines = [
            f"═══ Unitary Regulator — Step {report.step} ═══",
            f"  PLL Locked : {'YES' if report.pll_locked else 'NO'}",
            f"  Phase Error: {report.pll_phase_error:.6f}",
            "",
            "  Lyapunov Profile (λ per layer):",
        ]
        for i, lam in enumerate(report.lyapunov_profile):
            marker = " ← PAGE TIME" if i == 7 else ""
            phase = "scramble" if lam > 0 else "island  "
            lines.append(f"    Layer {i:2d}: λ={lam:+.4f}  [{phase}]{marker}")

        lines.append("")
        lines.append("  Topological Heat Map:")
        for idx, info in sorted(report.heatmap.items()):
            bar_len = int(info["island_strength"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"    Layer {idx:2d}: island={info['island_strength']:.3f} "
                f"  gap={info['spectral_gap']:.4f}  |{bar}|"
            )

        if report.casimir_diagnostics:
            lines.append("")
            lines.append("  Casimir Diagnostics:")
            for key, val in report.casimir_diagnostics.items():
                lines.append(f"    {key}: {val}")

        # --- Wormhole Gap Monitor (v1.2) ---
        if report.wormhole_gap is not None:
            lines.append("")
            status = "ALERT — BRIDGE COLLAPSING" if report.wormhole_alert else "STABLE"
            lines.append(f"  Wormhole Gap Monitor: Δλ = {report.wormhole_gap:.4f}  [{status}]")
            if report.bridge_diagnostics:
                bell = report.bridge_diagnostics.get("bell_correlation", "N/A")
                lines.append(f"    Bell Correlation: {bell}")
                lines.append(f"    Source → Sink: Layer {report.bridge_diagnostics.get('source_layer')} → Layer {report.bridge_diagnostics.get('sink_layer')}")
        elif report.bridge_diagnostics and not report.zeno_measurement_taken:
            lines.append("")
            lines.append("  Wormhole Gap Monitor: [SKIPPED — Poisson anti-resonance]")

        # --- Zeno Stabilization (v1.2-stable) ---
        if report.measurement_freq is not None:
            lines.append(f"  Zeno Measurement Freq: {report.measurement_freq:.3f}")
            lines.append(f"  Zeno Measurement Taken: {'YES' if report.zeno_measurement_taken else 'NO (Poisson skip)'}")

        text = "\n".join(lines)
        print(text)
        return text

    @property
    def history(self) -> List[RegulatorReport]:
        return list(self._reports)
