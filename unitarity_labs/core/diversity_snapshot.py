"""
core/diversity_snapshot.py — Drift Detection via Solo Inference Windows
========================================================================
v3.0.0-Singularity

Protects against coherent gradual semantic drift by periodically
running solo inference windows with the bridge disabled and comparing
solo vs bridged hidden states.

Behavior:
  Every 4096 tokens, force a 128-token solo inference window with
  bridge disabled. Compare solo vs bridged hidden states:

    ΔH = ||H_solo - H_bridged||_F

  Trigger condition:
    If mean(ΔH) < 0.08 × ||H||_F for two consecutive checkpoints,
    emit a "coherence collapse" warning.

Configurable but enabled by default in distributed active mode.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

SNAPSHOT_INTERVAL_TOKENS: int = 4096
SOLO_WINDOW_TOKENS: int = 128
COLLAPSE_THRESHOLD_RATIO: float = 0.08
CONSECUTIVE_TRIGGERS_REQUIRED: int = 2


@dataclass
class SnapshotCheckpoint:
    """Record of a single diversity snapshot checkpoint."""
    token_position: int
    delta_h_mean: float
    h_norm_mean: float
    ratio: float
    triggered: bool


class DiversitySnapshotMonitor:
    """Monitors for coherence collapse via solo inference windows.

    Parameters
    ----------
    interval_tokens : int
        Tokens between snapshot checkpoints (default 4096).
    solo_window_tokens : int
        Length of solo inference window (default 128).
    collapse_threshold : float
        Ratio threshold: if ΔH/||H|| < this for consecutive checkpoints,
        emit warning (default 0.08).
    enabled : bool
        Whether monitoring is active (default True).
    auto_reanchor : bool
        Whether to automatically trigger a re-anchor on collapse
        (default False — just warn).
    """

    def __init__(
        self,
        interval_tokens: int = SNAPSHOT_INTERVAL_TOKENS,
        solo_window_tokens: int = SOLO_WINDOW_TOKENS,
        collapse_threshold: float = COLLAPSE_THRESHOLD_RATIO,
        enabled: bool = True,
        auto_reanchor: bool = False,
    ):
        self.interval_tokens = interval_tokens
        self.solo_window_tokens = solo_window_tokens
        self.collapse_threshold = collapse_threshold
        self.enabled = enabled
        self.auto_reanchor = auto_reanchor

        self._token_counter: int = 0
        self._in_solo_window: bool = False
        self._solo_tokens_elapsed: int = 0

        # Hidden state accumulators for the solo window
        self._solo_states: List[torch.Tensor] = []
        self._bridged_states: List[torch.Tensor] = []

        self._checkpoints: List[SnapshotCheckpoint] = []
        self._consecutive_low_delta: int = 0
        self._collapse_warnings: int = 0
        self._reanchor_requested: bool = False

    @property
    def in_solo_window(self) -> bool:
        """True when the monitor is in a solo inference window."""
        return self._in_solo_window

    @property
    def should_disable_bridge(self) -> bool:
        """True when the bridge should be disabled for solo capture."""
        return self.enabled and self._in_solo_window

    @property
    def checkpoints(self) -> List[SnapshotCheckpoint]:
        return list(self._checkpoints)

    @property
    def collapse_warning_count(self) -> int:
        return self._collapse_warnings

    @property
    def reanchor_requested(self) -> bool:
        """True if auto-reanchor was triggered. Caller should reset."""
        return self._reanchor_requested

    def clear_reanchor_request(self) -> None:
        self._reanchor_requested = False

    def step(self) -> None:
        """Call after each token forward pass."""
        if not self.enabled:
            return

        self._token_counter += 1

        if self._in_solo_window:
            self._solo_tokens_elapsed += 1
            if self._solo_tokens_elapsed >= self.solo_window_tokens:
                self._finalize_checkpoint()
                self._in_solo_window = False
                self._solo_tokens_elapsed = 0
        else:
            if self._token_counter % self.interval_tokens == 0:
                self._begin_solo_window()

    def _begin_solo_window(self) -> None:
        """Enter solo inference window."""
        self._in_solo_window = True
        self._solo_tokens_elapsed = 0
        self._solo_states.clear()
        self._bridged_states.clear()
        logger.info(
            "Diversity snapshot: entering solo window at token %d",
            self._token_counter,
        )

    def record_states(
        self,
        h_solo: Optional[torch.Tensor],
        h_bridged: Optional[torch.Tensor],
    ) -> None:
        """Record hidden states during a solo window.

        Parameters
        ----------
        h_solo : Tensor or None
            Hidden state from solo (bridge-disabled) inference.
        h_bridged : Tensor or None
            Hidden state from most recent bridged inference
            (captured before the solo window started).
        """
        if not self._in_solo_window:
            return
        if h_solo is not None:
            self._solo_states.append(h_solo.detach())
        if h_bridged is not None:
            self._bridged_states.append(h_bridged.detach())

    def _finalize_checkpoint(self) -> None:
        """Compute ΔH and check trigger condition."""
        if not self._solo_states or not self._bridged_states:
            logger.warning("Diversity snapshot: no states captured in solo window")
            return

        # Stack and compute mean delta
        solo = torch.stack(self._solo_states)
        bridged = torch.stack(self._bridged_states[:len(self._solo_states)])

        # Truncate to matching length
        min_len = min(solo.shape[0], bridged.shape[0])
        solo = solo[:min_len]
        bridged = bridged[:min_len]

        delta_h = torch.norm(solo.float() - bridged.float(), p="fro").item()
        h_norm = torch.norm(solo.float(), p="fro").item()
        delta_mean = delta_h / max(min_len, 1)
        h_norm_mean = h_norm / max(min_len, 1)

        ratio = delta_mean / max(h_norm_mean, 1e-12)
        triggered = ratio < self.collapse_threshold

        checkpoint = SnapshotCheckpoint(
            token_position=self._token_counter,
            delta_h_mean=delta_mean,
            h_norm_mean=h_norm_mean,
            ratio=ratio,
            triggered=triggered,
        )
        self._checkpoints.append(checkpoint)

        logger.info(
            "Diversity snapshot checkpoint at token %d: "
            "ΔH_mean=%.6f, ||H||_mean=%.6f, ratio=%.4f, triggered=%s",
            self._token_counter, delta_mean, h_norm_mean, ratio, triggered,
        )

        if triggered:
            self._consecutive_low_delta += 1
        else:
            self._consecutive_low_delta = 0

        if self._consecutive_low_delta >= CONSECUTIVE_TRIGGERS_REQUIRED:
            self._collapse_warnings += 1
            logger.warning(
                "COHERENCE COLLAPSE WARNING #%d at token %d: "
                "mean(ΔH) < %.2f × ||H||_F for %d consecutive checkpoints. "
                "Solo and bridged hidden states are converging.",
                self._collapse_warnings,
                self._token_counter,
                self.collapse_threshold,
                CONSECUTIVE_TRIGGERS_REQUIRED,
            )
            if self.auto_reanchor:
                self._reanchor_requested = True
                logger.warning("Auto re-anchor requested.")
            # Reset counter so we warn again if it persists
            self._consecutive_low_delta = 0

        self._solo_states.clear()
        self._bridged_states.clear()

    def get_diagnostics(self) -> dict:
        """Return current monitoring state for dashboard/logs."""
        return {
            "enabled": self.enabled,
            "token_counter": self._token_counter,
            "in_solo_window": self._in_solo_window,
            "checkpoints_total": len(self._checkpoints),
            "collapse_warnings": self._collapse_warnings,
            "consecutive_low_delta": self._consecutive_low_delta,
            "last_checkpoint": (
                {
                    "token": self._checkpoints[-1].token_position,
                    "ratio": self._checkpoints[-1].ratio,
                    "triggered": self._checkpoints[-1].triggered,
                }
                if self._checkpoints
                else None
            ),
        }
