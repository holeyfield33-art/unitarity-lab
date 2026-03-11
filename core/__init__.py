# Holeyfield v1.2-stable — The Entanglement Bridge (Production)
# TMRP Session 18/19 Design Lock + DeepSeek Integration + LoRA Optimization

from .pll_monitor import PLLMonitor, SpectralAnomaly
from .horizons import PageCurveHook, singularity_warning, _lanczos_tridiagonal, _rayleigh_quotient_iteration
from .casimir_opt import CasimirOptimizer, rsvd
from .bridge import CrossLayerEntanglementHook, LoRABridgeAdapter, PROJECTION_NORM_MIN, PROJECTION_NORM_MAX
from .unitary_regulator import (
    wormhole_gap_alert, WORMHOLE_GAP_THRESHOLD,
    adaptive_measurement_freq, poisson_sampling_guard, enforce_projection_norm,
)

__all__ = [
    "PLLMonitor",
    "SpectralAnomaly",
    "PageCurveHook",
    "CasimirOptimizer",
    "CrossLayerEntanglementHook",
    "LoRABridgeAdapter",
    "singularity_warning",
    "wormhole_gap_alert",
    "WORMHOLE_GAP_THRESHOLD",
    "PROJECTION_NORM_MIN",
    "PROJECTION_NORM_MAX",
    "adaptive_measurement_freq",
    "poisson_sampling_guard",
    "enforce_projection_norm",
    "rsvd",
    "_lanczos_tridiagonal",
    "_rayleigh_quotient_iteration",
]
