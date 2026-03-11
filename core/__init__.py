# Holeyfield v1.2 — The Entanglement Bridge
# TMRP Session 18/19 Design Lock + DeepSeek Integration

from .pll_monitor import PLLMonitor, SpectralAnomaly
from .horizons import PageCurveHook, singularity_warning, _lanczos_tridiagonal, _rayleigh_quotient_iteration
from .casimir_opt import CasimirOptimizer, rsvd
from .bridge import CrossLayerEntanglementHook
from .unitary_regulator import wormhole_gap_alert, WORMHOLE_GAP_THRESHOLD

__all__ = [
    "PLLMonitor",
    "SpectralAnomaly",
    "PageCurveHook",
    "CasimirOptimizer",
    "CrossLayerEntanglementHook",
    "singularity_warning",
    "wormhole_gap_alert",
    "WORMHOLE_GAP_THRESHOLD",
    "rsvd",
    "_lanczos_tridiagonal",
    "_rayleigh_quotient_iteration",
]
