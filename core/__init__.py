# Holeyfield v1.7-unitary-link — Inter-Model ER=EPR (Production)
# TMRP Session 27 Mirror Integration | 128b << 12kb R_max | α=0.1 operating point

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
)
from .kill_switch import (
    ByzantineVoting, NodeStatus, NodeRecord,
    HARD_SEVER_THRESHOLD, GRACEFUL_THRESHOLD, READMIT_THRESHOLD, READMIT_EPOCHS,
)
from .universal_hook import UniversalHookWrapper
from .ghost_layer import RecursiveMirror
from .dashboard import HeartbeatDashboard
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
]
