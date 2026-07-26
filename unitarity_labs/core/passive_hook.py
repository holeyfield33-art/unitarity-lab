"""
core/passive_hook.py — Clean passive-mode telemetry hook.
============================================================
Wraps a passive-mode :class:`~core.universal_hook.UniversalHookWrapper`
and exposes exactly the three fields an external caller needs:
``zeta_raw``, ``spectral_gap``, and ``flagged``.

``flagged`` is not hardcoded: each call feeds the wrapper's current
``spectral_gap`` into a real calibrated rupture detector from the VAR
package (``var_spectral.detector.SpectralRuptureDetector`` — median/MAD
baseline with hysteresis, see https://github.com/holeyfield33-art/VAR).
This is the "depend on VAR, don't duplicate" module for unitarity-lab:
the detector's own calibration/rupture state machine is reused as-is
rather than reimplemented here.

VAR is an optional dependency, so importing this module requires
``pip install 'unitarity-labs[spectral]'``. Note that VAR's import
package is ``var_spectral``, not ``var`` — the latter is an unrelated
project on PyPI.
"""

from __future__ import annotations

from typing import Optional, TypedDict

try:
    from var_spectral.detector import RuptureState, SpectralRuptureDetector
except ImportError as e:  # pragma: no cover - exercised only without VAR
    raise ImportError(
        "PassiveTelemetryHook requires the VAR package. "
        "Install with: pip install 'unitarity-labs[spectral]'\n"
        "Note: VAR's import package is 'var_spectral' and its distribution "
        "is 'var-spectral'. Do not `pip install var` — that is an unrelated "
        "project (portfolio Value-at-Risk) which shadows this one."
    ) from e

from .universal_hook import UniversalHookWrapper


class PassiveTelemetry(TypedDict):
    zeta_raw: float
    spectral_gap: float
    flagged: bool


class PassiveTelemetryHook:
    """Read-only telemetry over a passive-mode ``UniversalHookWrapper``.

    Construction does not mutate the wrapper or its model in any way —
    ``read()`` only calls ``wrapper.get_metrics()`` (already read-only)
    and forwards the resulting ``spectral_gap`` into its own rupture
    detector instance.
    """

    def __init__(
        self,
        wrapper: UniversalHookWrapper,
        rupture_detector: Optional[SpectralRuptureDetector] = None,
    ) -> None:
        if wrapper.mode != "passive":
            raise ValueError(
                f"PassiveTelemetryHook requires a passive-mode wrapper, got mode={wrapper.mode!r}"
            )
        self.wrapper = wrapper
        self.rupture_detector = rupture_detector or SpectralRuptureDetector()

    def read(self) -> PassiveTelemetry:
        """One telemetry sample: ``{zeta_raw, spectral_gap, flagged}``.

        Call once per forward pass (after the model has run) so
        ``spectral_gap`` reflects the latest source-layer activation.
        """
        metrics = self.wrapper.get_metrics()
        zeta_raw = float(metrics["zeta_raw"])
        spectral_gap = float(metrics["spectral_gap"])
        self.rupture_detector.update(spectral_gap)
        flagged = self.rupture_detector.state is RuptureState.RUPTURED
        return {"zeta_raw": zeta_raw, "spectral_gap": spectral_gap, "flagged": flagged}
