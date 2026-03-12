"""
precision_projector.py — Precision Alignment Protocol (v3.0.0-Singularity)
============================================================================
Handles precision mismatch between nodes with different quantization
(INT4, FP8, BF16, FP32). Provides:

  - ``PrecisionClass`` enum for node precision classification.
  - ``DequantAdapter`` — lightweight trainable linear projector for
    casting between incompatible precision classes.
  - ``PROJECTOR_REGISTRY`` — maps ``(src, tgt)`` precision pairs to
    projector factory callables.
  - ``add_dither`` — stochastic rounding noise to preserve low-bit
    information during BF16 canonical cast.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
# Precision Classes
# ======================================================================

class PrecisionClass(str, Enum):
    INT4 = "INT4"
    FP8_E4M3 = "FP8_E4M3"
    BF16 = "BF16"
    FP32 = "FP32"


# Canonical gossip dtype — all cross-node messages are cast to this.
CANONICAL_DTYPE = torch.bfloat16


# ======================================================================
# Dithering
# ======================================================================

def add_dither(x: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """Add stochastic rounding noise scaled to the target bit-width.

    This preserves low-bit information when casting to a lower
    precision by injecting uniform noise at the quantization step size.

    Parameters
    ----------
    x : Tensor
        Input tensor (any dtype).
    bits : int
        Target precision in bits (e.g. 16 for BF16).

    Returns
    -------
    Tensor in ``CANONICAL_DTYPE`` with dithering applied.
    """
    x_float = x.float()
    scale = x_float.abs().max().clamp(min=1e-12)
    step = scale / (2 ** (bits - 1))
    noise = torch.empty_like(x_float).uniform_(-0.5, 0.5) * step
    return (x_float + noise).to(CANONICAL_DTYPE)


# ======================================================================
# DequantAdapter
# ======================================================================

class DequantAdapter(nn.Module):
    """Lightweight trainable linear projector for precision conversion.

    A small offline-tuned linear layer that learns to map from the
    canonical BF16 gossip representation to/from a node's local
    precision, compensating for quantization artefacts.

    Parameters
    ----------
    dim : int
        Feature dimension of the tensors being projected.
    bias : bool
        Include learnable bias (default True).
    """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)
        # Identity init so adapter is pass-through before tuning
        nn.init.eye_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float()).to(x.dtype)


# ======================================================================
# Projector Registry
# ======================================================================

def _make_adapter(dim: int) -> DequantAdapter:
    """Factory for a default DequantAdapter at the given dimension."""
    return DequantAdapter(dim)


# Maps (source_precision, target_precision) -> factory(dim) -> DequantAdapter.
# Same-precision pairs communicate directly and don't need a projector.
PROJECTOR_REGISTRY: Dict[Tuple[PrecisionClass, PrecisionClass], Callable[[int], DequantAdapter]] = {
    (PrecisionClass.INT4, PrecisionClass.BF16): _make_adapter,
    (PrecisionClass.BF16, PrecisionClass.INT4): _make_adapter,
    (PrecisionClass.INT4, PrecisionClass.FP32): _make_adapter,
    (PrecisionClass.FP32, PrecisionClass.INT4): _make_adapter,
    (PrecisionClass.INT4, PrecisionClass.FP8_E4M3): _make_adapter,
    (PrecisionClass.FP8_E4M3, PrecisionClass.INT4): _make_adapter,
    (PrecisionClass.FP8_E4M3, PrecisionClass.BF16): _make_adapter,
    (PrecisionClass.BF16, PrecisionClass.FP8_E4M3): _make_adapter,
    (PrecisionClass.FP8_E4M3, PrecisionClass.FP32): _make_adapter,
    (PrecisionClass.FP32, PrecisionClass.FP8_E4M3): _make_adapter,
    (PrecisionClass.BF16, PrecisionClass.FP32): _make_adapter,
    (PrecisionClass.FP32, PrecisionClass.BF16): _make_adapter,
}


def get_projector(
    src: PrecisionClass, tgt: PrecisionClass, dim: int,
) -> Optional[DequantAdapter]:
    """Look up and instantiate a projector for the given precision pair.

    Returns None if the pair can communicate directly (same precision)
    or if no projector is registered.
    """
    if src == tgt:
        return None
    factory = PROJECTOR_REGISTRY.get((src, tgt))
    if factory is None:
        return None
    return factory(dim)


def has_projector(src: PrecisionClass, tgt: PrecisionClass) -> bool:
    """Check if a projector exists for the given precision pair."""
    return src == tgt or (src, tgt) in PROJECTOR_REGISTRY
