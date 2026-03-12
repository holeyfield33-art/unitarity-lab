"""
labs/topology_metrics.py — Experimental topology-derived metrics.
=================================================================
v3.0.0-Singularity

**EXPERIMENTAL**: This module provides experimental topology-based
metrics for research use. These metrics are not validated for
production and their definitions may change.

Provides wrappers around internal topology diagnostics (spectral gap,
Betti numbers, Lyapunov profiles) for use in experimental analysis
pipelines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core.unitary_regulator import compute_topological_heatmap
from core.casimir_opt import estimate_betti_0


def spectral_gap_from_activations(
    activations: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """Compute per-layer spectral gap from captured activations.

    Parameters
    ----------
    activations : dict[int, Tensor]
        Layer index → activation tensor.

    Returns
    -------
    dict[int, float]
        Layer index → spectral gap.
    """
    heatmap = compute_topological_heatmap(activations)
    return {idx: vals["spectral_gap"] for idx, vals in heatmap.items()}


def betti_0_from_weights(
    model: torch.nn.Module,
    threshold: float = 0.1,
) -> Dict[str, int]:
    """Estimate Betti-0 (connected components) for each parameter tensor.

    Parameters
    ----------
    model : nn.Module
    threshold : float
        Cosine similarity threshold for adjacency.

    Returns
    -------
    dict[str, int]
        Parameter name → estimated Betti-0.
    """
    result: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            result[name] = estimate_betti_0(param.data, threshold)
    return result


def activation_entropy_profile(
    activations: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """Compute per-layer activation entropy.

    Parameters
    ----------
    activations : dict[int, Tensor]

    Returns
    -------
    dict[int, float]
        Layer index → activation entropy.
    """
    heatmap = compute_topological_heatmap(activations)
    return {idx: vals["entropy"] for idx, vals in heatmap.items()}
