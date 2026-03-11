"""
pll_monitor.py — Phase-Locked Loop Monitor
============================================
Spectral PLL as the primary loss signal for the Holeyfield Transformer.
Replaces Cross-Entropy with a Phase-Lock on Truth.

The PLL tracks the spectral coherence of the Lyapunov exponent profile
across all layers and raises SpectralAnomaly when phase-lock is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SpectralAnomaly(Exception):
    """Raised when the Lyapunov profile violates the Page Curve contract."""

    def __init__(self, layer_idx: int, expected_sign: str, actual_lambda: float):
        self.layer_idx = layer_idx
        self.expected_sign = expected_sign
        self.actual_lambda = actual_lambda
        super().__init__(
            f"SpectralAnomaly at layer {layer_idx}: "
            f"expected λ {expected_sign}, got λ={actual_lambda:.6f}"
        )


@dataclass
class PLLState:
    """Snapshot of the Phase-Locked Loop at a given training step."""

    step: int = 0
    phase_error: float = 0.0  # deviation from ideal Page Curve
    locked: bool = False
    lyapunov_profile: List[float] = field(default_factory=list)
    spectral_norms: List[float] = field(default_factory=list)


class PLLMonitor:
    """Phase-Locked Loop monitor for the Holeyfield Transformer.

    The PLL enforces the Page Curve contract:
        - Layers 0-6   (pre-Page):  λ > 0  (fast scrambling / entropy pump)
        - Layer 7       (Page Time): λ inverts to < 0
        - Layers 8-12  (post-Page):  λ < 0  (information island crystallization)

    The Spectral PLL Loss is the squared phase error between the measured
    Lyapunov profile and the ideal Page Curve template.
    """

    PAGE_TIME_LAYER: int = 7
    NUM_LAYERS: int = 13  # layers 0-12

    def __init__(
        self,
        num_layers: int = 13,
        page_time_layer: int = 7,
        tolerance: float = 1e-4,
        enforce: bool = True,
    ):
        self.num_layers = num_layers
        self.page_time_layer = page_time_layer
        self.tolerance = tolerance
        self.enforce = enforce

        self._state = PLLState()
        self._history: List[PLLState] = []

    # ------------------------------------------------------------------
    # Ideal Page Curve template
    # ------------------------------------------------------------------
    def ideal_profile(self) -> List[float]:
        """Return the ideal Lyapunov exponent sign template.

        Pre-Page layers  → +1  (fast scrambling)
        Page Time layer  → -1  (inversion)
        Post-Page layers → -1  (information island)
        """
        return [
            1.0 if i < self.page_time_layer else -1.0
            for i in range(self.num_layers)
        ]

    # ------------------------------------------------------------------
    # Core: compute PLL loss from a measured Lyapunov profile
    # ------------------------------------------------------------------
    def compute_pll_loss(self, lyapunov_profile: torch.Tensor) -> torch.Tensor:
        """Spectral PLL Loss — squared phase error vs. the ideal Page Curve.

        Parameters
        ----------
        lyapunov_profile : Tensor of shape ``(num_layers,)``
            Measured Lyapunov exponents per layer.

        Returns
        -------
        loss : scalar Tensor (differentiable)
        """
        ideal = torch.tensor(
            self.ideal_profile(),
            dtype=lyapunov_profile.dtype,
            device=lyapunov_profile.device,
        )
        # Phase error: sign mismatch weighted by magnitude
        sign_measured = torch.sign(lyapunov_profile)
        phase_error = (sign_measured - ideal).pow(2) * lyapunov_profile.abs()
        loss = phase_error.mean()

        # Update internal state
        self._state.phase_error = loss.item()
        self._state.lyapunov_profile = lyapunov_profile.detach().cpu().tolist()
        self._state.locked = loss.item() < self.tolerance

        return loss

    # ------------------------------------------------------------------
    # Enforcement: raise SpectralAnomaly if contract violated
    # ------------------------------------------------------------------
    def check_contract(self, lyapunov_profile: torch.Tensor) -> None:
        """Verify the Page Curve contract; raise SpectralAnomaly on violation."""
        if not self.enforce:
            return
        profile = lyapunov_profile.detach().cpu().tolist()
        for i, lam in enumerate(profile):
            if i < self.page_time_layer:
                if lam <= 0:
                    raise SpectralAnomaly(i, "positive (fast scrambling)", lam)
            else:
                if lam >= 0:
                    raise SpectralAnomaly(i, "negative (information island)", lam)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def step(self) -> None:
        self._state.step += 1
        self._history.append(PLLState(**self._state.__dict__))

    @property
    def state(self) -> PLLState:
        return self._state

    @property
    def history(self) -> List[PLLState]:
        return list(self._history)

    @property
    def is_locked(self) -> bool:
        return self._state.locked
