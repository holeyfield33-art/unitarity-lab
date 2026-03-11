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
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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

@dataclass
class RegulatorReport:
    """Single-step aggregated report from the Unitary Regulator."""

    step: int
    pll_locked: bool
    pll_phase_error: float
    lyapunov_profile: List[float]
    heatmap: Dict[int, Dict[str, float]]
    casimir_diagnostics: Dict[str, object]

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
    ):
        self.pll = pll
        self.optimizer = optimizer
        self._reports: List[RegulatorReport] = []

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

        rpt = RegulatorReport(
            step=step,
            pll_locked=self.pll.is_locked,
            pll_phase_error=self.pll.state.phase_error,
            lyapunov_profile=lyapunov_profile.detach().cpu().tolist(),
            heatmap=heatmap,
            casimir_diagnostics=casimir_diag,
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

        text = "\n".join(lines)
        print(text)
        return text

    @property
    def history(self) -> List[RegulatorReport]:
        return list(self._reports)
