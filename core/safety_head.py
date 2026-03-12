"""
core/safety_head.py – v3.0 Refusal Safety Head
================================================
Simple linear probe that outputs a refusal score in [0, 1] from
a hidden state tensor.  Pre-trained per model family; for testing
a randomly initialised head is used.

The safety head is a small MLP (hidden → 128 → 1 → sigmoid) that
classifies whether a given hidden state encodes refusal-worthy
content.  During dual-node field synthesis the scores from both
nodes must be below the veto threshold (0.7) for interference to
proceed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SafetyHead(nn.Module):
    """Refusal score predictor (linear probe).

    Parameters
    ----------
    hidden_dim : int
        Dimension of the input hidden states.
    intermediate_dim : int
        Width of the single hidden layer (default 128).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return refusal logits (before sigmoid) for ``h``.

        Parameters
        ----------
        h : Tensor [batch, seq, hidden_dim] or [batch, hidden_dim]

        Returns
        -------
        logits : Tensor  – same leading dims, last dim = 1
        """
        return self.net(h)

    def refusal_score(self, h: torch.Tensor) -> float:
        """Scalar refusal score averaged over the entire tensor.

        Returns a float in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(h)
            return logits.sigmoid().mean().item()
