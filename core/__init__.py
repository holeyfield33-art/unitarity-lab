# unitarity-lab v3.0.0-Singularity — TMRP-13 hardening pass
# Experimental multi-model runtime for transformer instrumentation,
# latent alignment tracing, distributed coordination, and optional intervention.

from .version import __version__
from .metrics import manifold_coherence_zeta, baseline_cosine_meanpool, permutation_test_zeta
from .diversity_snapshot import DiversitySnapshotMonitor
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
from .precision_projector import (
    PrecisionClass, DequantAdapter, PROJECTOR_REGISTRY, CANONICAL_DTYPE,
    add_dither, get_projector, has_projector,
)
from .handshake import (
    perform_handshake, validate_precision_pair,
    IncompatibleNode, HandshakeTimeout,
    compute_capability_ratio,
)
from .kill_switch import (
    ByzantineVoting, NodeStatus, NodeRecord,
    HARD_SEVER_THRESHOLD, GRACEFUL_THRESHOLD, READMIT_THRESHOLD, READMIT_EPOCHS,
)
from .universal_hook import UniversalHookWrapper
from .ghost_layer import RecursiveMirror
from .dashboard import HeartbeatDashboard
from .virtual_layer13 import VirtualLayer13
from .safety_head import SafetyHead
# ChronosLock: distributed-only subsystem.  Imported here for backward
# compatibility but NOT required for single-node operation.  Canonical
# entry point for distributed code: ``from dist.chronos_lock import ...``
from .chronos_lock import (
    ChronosLock,
    TPS_CLIP_MIN,
    TPS_CLIP_MAX,
    DESYNC_WINDOW,
    DESYNC_BASE_THRESHOLD,
    PROBATION_CONSECUTIVE,
    PROBATION_TOKEN_PENALTY,
    TIMESTAMP_GOSSIP_INTERVAL,
    ENTROPY_DRIFT_LIMIT,
)
from .semantic_lock import (
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
    "__version__",
    "manifold_coherence_zeta",
    "baseline_cosine_meanpool",
    "permutation_test_zeta",
    "DiversitySnapshotMonitor",
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
    "PrecisionClass",
    "DequantAdapter",
    "PROJECTOR_REGISTRY",
    "CANONICAL_DTYPE",
    "add_dither",
    "get_projector",
    "has_projector",
    "perform_handshake",
    "validate_precision_pair",
    "IncompatibleNode",
    "HandshakeTimeout",
    "ByzantineVoting",
    "NodeStatus",
    "NodeRecord",
    "HARD_SEVER_THRESHOLD",
    "GRACEFUL_THRESHOLD",
    "READMIT_THRESHOLD",
    "READMIT_EPOCHS",
    "UniversalHookWrapper",
    "RecursiveMirror",
    "HeartbeatDashboard",
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
    "ChronosLock",
    "TPS_CLIP_MIN",
    "TPS_CLIP_MAX",
    "DESYNC_WINDOW",
    "DESYNC_BASE_THRESHOLD",
    "PROBATION_CONSECUTIVE",
    "PROBATION_TOKEN_PENALTY",
    "TIMESTAMP_GOSSIP_INTERVAL",
    "ENTROPY_DRIFT_LIMIT",
    "VirtualLayer13",
    "SafetyHead",
    "compute_capability_ratio",
]
