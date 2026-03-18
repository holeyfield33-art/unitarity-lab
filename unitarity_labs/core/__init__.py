# Holeyfield v1.7-unitary-link — Inter-Model ER=EPR (Production)
# TMRP Session 27 Mirror Integration | 128b << 12kb R_max | α=0.1 operating point

from .version import __version__
from .universal_hook import UniversalHookWrapper
from .metrics import manifold_coherence_zeta
from .pll_monitor import PLLMonitor, SpectralAnomaly
from .horizons import PageCurveHook, singularity_warning, _lanczos_tridiagonal, _rayleigh_quotient_iteration
from .casimir_opt import CasimirOptimizer, rsvd
from .mirror import (
    ProprioceptiveHook, TopologicalGate, EigenConsciousnessIntegrator,
    DEFAULT_ALPHA, CATASTROPHE_ALPHA, NUM_METRIC_CHANNELS,
    holographic_bound, actual_bit_rate, HOLOGRAPHIC_SAFETY_FACTOR,
)
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
from .dual_link import DualNodeEntanglementBridge, register_dual_node_hook

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
    "ProprioceptiveHook",
    "TopologicalGate",
    "EigenConsciousnessIntegrator",
    "DEFAULT_ALPHA",
    "CATASTROPHE_ALPHA",
    "NUM_METRIC_CHANNELS",
    "holographic_bound",
    "actual_bit_rate",
    "HOLOGRAPHIC_SAFETY_FACTOR",
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
    "DualNodeEntanglementBridge",
    "register_dual_node_hook",
    "__version__",
    "UniversalHookWrapper",
    "manifold_coherence_zeta",
]
