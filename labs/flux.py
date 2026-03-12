"""
labs/flux.py — Compatibility shim.
Canonical location: core/flux.py (to be migrated).

**EXPERIMENTAL**: This module is a research prototype.
"""

from core.flux import (
    HawkingFluxGovernor,
    HAWKING_DECAY_RATE,
    batch_goe,
    batch_expm,
    select_staggered_heads,
    STAGGER_FRACTION,
    TAYLOR_DIM_THRESHOLD,
    TAYLOR_ERROR_GUARD,
)

__all__ = [
    "HawkingFluxGovernor",
    "HAWKING_DECAY_RATE",
    "batch_goe",
    "batch_expm",
    "select_staggered_heads",
    "STAGGER_FRACTION",
    "TAYLOR_DIM_THRESHOLD",
    "TAYLOR_ERROR_GUARD",
]
