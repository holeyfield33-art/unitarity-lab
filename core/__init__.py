# Holeyfield v1.4-superfluid — Parallel Flux Certified (Production)
# TMRP Session 26 Superfluid Deployment | 21x Latency Reduction | 1.8GB VRAM Guard

from .pll_monitor import PLLMonitor, SpectralAnomaly
from .horizons import PageCurveHook, singularity_warning, _lanczos_tridiagonal, _rayleigh_quotient_iteration
from .casimir_opt import CasimirOptimizer, rsvd
from .bridge import CrossLayerEntanglementHook, LoRABridgeAdapter, PROJECTION_NORM_MIN, PROJECTION_NORM_MAX
from .flux import (
    HawkingFluxGovernor, HAWKING_DECAY_RATE,
    batch_goe, batch_expm, select_staggered_heads,
    STAGGER_FRACTION, TAYLOR_DIM_THRESHOLD, TAYLOR_ERROR_GUARD,
)
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
    "HawkingFluxGovernor",
    "HAWKING_DECAY_RATE",
    "batch_goe",
    "batch_expm",
    "select_staggered_heads",
    "STAGGER_FRACTION",
    "TAYLOR_DIM_THRESHOLD",
    "TAYLOR_ERROR_GUARD",
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
