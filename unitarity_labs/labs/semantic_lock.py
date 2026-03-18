"""
labs/semantic_lock.py — Compatibility shim.
Canonical location: core/semantic_lock.py (to be migrated).

**EXPERIMENTAL**: This module is a research prototype.
"""

from unitarity_labs.core.semantic_lock import (
    SemanticLockController,
    SemanticModulator,
    AnchorConsensusGossip,
    NonceCommitProtocol,
    NonceCommitState,
    semantic_anchor_init,
    compute_alpha_sem,
    compute_alpha_sem_ensemble,
    semantic_modulation,
    holographic_semantic_shard_encode,
    holographic_semantic_shard_decode,
    validate_shard_integrity,
    SEM_PROJECTION_DIM,
    ANCHOR_DRIFT_THRESHOLD,
    ALPHA_SEM_FULL_BRIDGE,
    ALPHA_SEM_PARTIAL_FLOOR,
    ALPHA_SEM_BYZANTINE_THRESHOLD,
    FLASH_VETO_DELTA,
    ANCHOR_GOSSIP_INTERVAL,
    ANCHOR_FREEZE_TOKENS,
)

__all__ = [
    "SemanticLockController",
    "SemanticModulator",
    "AnchorConsensusGossip",
    "NonceCommitProtocol",
    "NonceCommitState",
    "semantic_anchor_init",
    "compute_alpha_sem",
    "compute_alpha_sem_ensemble",
    "semantic_modulation",
    "holographic_semantic_shard_encode",
    "holographic_semantic_shard_decode",
    "validate_shard_integrity",
    "SEM_PROJECTION_DIM",
    "ANCHOR_DRIFT_THRESHOLD",
    "ALPHA_SEM_FULL_BRIDGE",
    "ALPHA_SEM_PARTIAL_FLOOR",
    "ALPHA_SEM_BYZANTINE_THRESHOLD",
    "FLASH_VETO_DELTA",
    "ANCHOR_GOSSIP_INTERVAL",
    "ANCHOR_FREEZE_TOKENS",
]
