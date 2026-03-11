# Holeyfield v1.1 — Unitary Core
# TMRP Session 18 Design Lock + DeepSeek Integration

from .pll_monitor import PLLMonitor, SpectralAnomaly
from .horizons import PageCurveHook, singularity_warning, _lanczos_tridiagonal, _rayleigh_quotient_iteration
from .casimir_opt import CasimirOptimizer, rsvd

__all__ = [
    "PLLMonitor",
    "SpectralAnomaly",
    "PageCurveHook",
    "CasimirOptimizer",
    "singularity_warning",
    "rsvd",
    "_lanczos_tridiagonal",
    "_rayleigh_quotient_iteration",
]
