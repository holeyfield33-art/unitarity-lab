"""
conftest.py — Shared pytest fixtures for the Holeyfield test suite.
====================================================================
Provides common model definitions and fixtures used across
test_criticality.py, test_mirror.py, and test_uncertainty.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ======================================================================
# Shared Model Definitions
# ======================================================================

class ToyTransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ToyTransformer(nn.Module):
    """13-layer toy transformer matching the Holeyfield layer layout."""

    def __init__(self, d_model: int = 64, num_layers: int = 13):
        super().__init__()
        self.layers = nn.ModuleList(
            [ToyTransformerLayer(d_model) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ======================================================================
# Shared Fixtures
# ======================================================================

@pytest.fixture
def toy_transformer():
    """Standard 13-layer toy transformer at d_model=64."""
    return ToyTransformer(d_model=64, num_layers=13)


@pytest.fixture
def toy_model():
    """Alias for toy_transformer (used in test_criticality and test_mirror)."""
    return ToyTransformer(d_model=64, num_layers=13)
