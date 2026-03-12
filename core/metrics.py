"""
core/metrics.py — Manifold Coherence ζ and baseline metrics.
=============================================================
Provides the canonical metric definitions for unitarity-lab v3.0.0-Singularity.

ζ is a flattened cosine similarity between activations. It is not a measure
of quantum non-locality, topological coherence, or Bell inequality violation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def manifold_coherence_zeta(
    source: torch.Tensor,
    sink: torch.Tensor,
) -> float:
    """Compute Manifold Coherence ζ between source and sink activations.

    Definition:
        ζ = cosine_similarity(flatten(H_source), flatten(H_sink))

    Both tensors are flattened to 1-D before computing cosine similarity.
    If shapes differ, the smaller tensor is zero-padded to match.

    Parameters
    ----------
    source : Tensor
        Source activation tensor (any shape).
    sink : Tensor
        Sink activation tensor (any shape).

    Returns
    -------
    zeta : float
        Cosine similarity in [-1, 1].
    """
    s_flat = source.detach().float().flatten()
    k_flat = sink.detach().float().flatten()

    # Handle shape mismatch by zero-padding the shorter tensor
    if s_flat.shape[0] != k_flat.shape[0]:
        max_len = max(s_flat.shape[0], k_flat.shape[0])
        if s_flat.shape[0] < max_len:
            s_flat = F.pad(s_flat, (0, max_len - s_flat.shape[0]))
        else:
            k_flat = F.pad(k_flat, (0, max_len - k_flat.shape[0]))

    return F.cosine_similarity(s_flat.unsqueeze(0), k_flat.unsqueeze(0)).item()


def baseline_cosine_meanpool(
    source: torch.Tensor,
    sink: torch.Tensor,
) -> float:
    """Baseline cosine similarity using mean-pooled activations.

    Mean-pools each tensor across all dimensions except the last,
    then computes cosine similarity on the resulting vectors.

    Parameters
    ----------
    source : Tensor
        Source activation tensor (at least 1-D).
    sink : Tensor
        Sink activation tensor (at least 1-D).

    Returns
    -------
    score : float
        Cosine similarity in [-1, 1].
    """
    s = source.detach().float()
    k = sink.detach().float()

    # Mean-pool to 1-D vectors
    while s.dim() > 1:
        s = s.mean(dim=0)
    while k.dim() > 1:
        k = k.mean(dim=0)

    # Handle dimension mismatch
    if s.shape[0] != k.shape[0]:
        min_len = min(s.shape[0], k.shape[0])
        s = s[:min_len]
        k = k[:min_len]

    return F.cosine_similarity(s.unsqueeze(0), k.unsqueeze(0)).item()


def permutation_test_zeta(
    source: torch.Tensor,
    sink: torch.Tensor,
    n_perm: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Permutation test for Manifold Coherence ζ significance.

    Computes the observed ζ, then generates n_perm null-distribution
    samples by randomly permuting the flattened sink tensor. Returns
    both the observed ζ and the two-sided p-value.

    Parameters
    ----------
    source : Tensor
        Source activation tensor.
    sink : Tensor
        Sink activation tensor.
    n_perm : int
        Number of permutations (default 10000).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    (zeta, p_value) : tuple of float
        zeta: the observed Manifold Coherence ζ.
        p_value: fraction of permuted samples with |ζ_perm| >= |ζ_obs|.
    """
    s_flat = source.detach().float().flatten()
    k_flat = sink.detach().float().flatten()

    # Align sizes
    if s_flat.shape[0] != k_flat.shape[0]:
        max_len = max(s_flat.shape[0], k_flat.shape[0])
        if s_flat.shape[0] < max_len:
            s_flat = F.pad(s_flat, (0, max_len - s_flat.shape[0]))
        else:
            k_flat = F.pad(k_flat, (0, max_len - k_flat.shape[0]))

    zeta_obs = F.cosine_similarity(s_flat.unsqueeze(0), k_flat.unsqueeze(0)).item()

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    count_extreme = 0
    for _ in range(n_perm):
        perm_idx = torch.randperm(k_flat.shape[0], generator=gen)
        k_perm = k_flat[perm_idx]
        zeta_perm = F.cosine_similarity(
            s_flat.unsqueeze(0), k_perm.unsqueeze(0)
        ).item()
        if abs(zeta_perm) >= abs(zeta_obs):
            count_extreme += 1

    p_value = count_extreme / n_perm
    return zeta_obs, p_value
