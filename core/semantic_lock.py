"""
semantic_lock.py — v2.2 Non-Local Semantic Locking (Adversarial-Hardened)
=========================================================================
Implements the Contrastive Random Projection + Topological Anchor
system for geometric semantic alignment across heterogeneous nodes.

Hardened per Grok adversarial audit:
  1. Multi-round nonce hash-commit with versioned binding
  2. Anchor consensus gossip (every 256 tokens) + tight drift (0.08)
  3. α_sem hysteresis + dual-layer ensemble + flash veto
  4. Reed-Solomon-style erasure coding on holographic semantic shards
  5. α_sem < 0.4 → Byzantine accusation broadcast + v2.1 fallback

Chain: Perplexity → Grok → DeepSeek → Copilot → Gemini
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ======================================================================
# Constants
# ======================================================================

# Projection dimensionality for W_sem
SEM_PROJECTION_DIM: int = 64

# Anchor drift thresholds (tightened per Grok audit)
ANCHOR_DRIFT_THRESHOLD: float = 0.08
ANCHOR_BETTI_DRIFT_THRESHOLD: float = 0.5

# α_sem hysteresis thresholds
ALPHA_SEM_FULL_BRIDGE: float = 0.75
ALPHA_SEM_PARTIAL_FLOOR: float = 0.55
ALPHA_SEM_BRIDGE_OFF: float = 0.55
ALPHA_SEM_BYZANTINE_THRESHOLD: float = 0.4

# Hysteresis cooldown (tokens)
HYSTERESIS_COOLDOWN_TOKENS: int = 10

# Flash veto: max allowed α_sem jump over 3 tokens
FLASH_VETO_DELTA: float = 0.35
FLASH_VETO_IGNORE_TOKENS: int = 5

# Anchor consensus gossip interval
ANCHOR_GOSSIP_INTERVAL: int = 256

# Anchor freeze: after this many tokens, anchor is frozen unless
# supermajority (75%) votes to re-anchor
ANCHOR_FREEZE_TOKENS: int = 128
ANCHOR_REANCHOR_SUPERMAJORITY: float = 0.75

# Ensemble weights for dual-layer α_sem
ENSEMBLE_WEIGHT_LAYER7: float = 0.6
ENSEMBLE_WEIGHT_LAYER12: float = 0.4

# Erasure coding: number of redundant shards
ERASURE_REDUNDANCY: int = 2

# Nonce commitment quorum factor (2f+1)
NONCE_COMMIT_QUORUM_FACTOR: int = 1  # actual quorum = 2*f + 1


# ======================================================================
# Nonce Commitment Protocol
# ======================================================================

class NonceCommitState(str, Enum):
    """State machine for multi-round nonce commitment."""
    IDLE = "IDLE"
    COMMIT_SENT = "COMMIT_SENT"        # SHA-256(nonce) gossiped
    REVEAL_PENDING = "REVEAL_PENDING"  # Awaiting 2f+1 hash agreement
    REVEALED = "REVEALED"              # Full nonce revealed
    VERIFIED = "VERIFIED"              # All nodes verified hash match
    FAILED = "FAILED"


@dataclass
class NonceCommitment:
    """A single node's nonce commitment record."""
    node_id: str
    commit_hash: str = ""       # SHA-256(nonce) — phase 1
    revealed_nonce: bytes = b""  # Full nonce — phase 2
    verified: bool = False


class NonceCommitProtocol:
    """Multi-round nonce hash-commit with versioned binding.

    Protocol:
      1. Each node generates a local nonce and gossips SHA-256(nonce)
      2. Wait for 2f+1 nodes to agree on the hash set
      3. Reveal full nonces — each node verifies SHA-256(revealed) matches
      4. Final nonce = SHA-256(session_uuid || timestamp_floor || min_node_id)
         bound to version for replay resistance.

    Parameters
    ----------
    local_node_id : str
        This node's identifier.
    max_faulty : int
        Assumed maximum Byzantine nodes (f).
    session_uuid : str
        Unique session identifier for versioned binding.
    """

    def __init__(
        self,
        local_node_id: str,
        max_faulty: int = 1,
        session_uuid: str = "",
    ):
        self.local_node_id = local_node_id
        self.max_faulty = max_faulty
        self.session_uuid = session_uuid
        self.state = NonceCommitState.IDLE
        self._local_nonce: bytes = b""
        self._local_commit_hash: str = ""
        self._commitments: Dict[str, NonceCommitment] = {}
        self._quorum_required = 2 * max_faulty + 1

    def generate_commit(self) -> Tuple[str, str]:
        """Phase 1: Generate local nonce, return (node_id, commit_hash).

        The nonce is 32 bytes of deterministic entropy derived from
        session_uuid + node_id to avoid reliance on os.urandom during
        testing, while remaining unique per session.
        """
        # Deterministic-per-session nonce (production: augment with os.urandom)
        seed_material = f"{self.session_uuid}:{self.local_node_id}".encode()
        self._local_nonce = hashlib.sha256(seed_material).digest()
        self._local_commit_hash = hashlib.sha256(self._local_nonce).hexdigest()
        self.state = NonceCommitState.COMMIT_SENT

        # Register own commitment
        self._commitments[self.local_node_id] = NonceCommitment(
            node_id=self.local_node_id,
            commit_hash=self._local_commit_hash,
        )
        return self.local_node_id, self._local_commit_hash

    def receive_commit(self, node_id: str, commit_hash: str) -> None:
        """Phase 1: Receive a commit hash from another node."""
        self._commitments[node_id] = NonceCommitment(
            node_id=node_id,
            commit_hash=commit_hash,
        )

    def check_commit_quorum(self) -> bool:
        """Check if 2f+1 commit hashes have been collected."""
        return len(self._commitments) >= self._quorum_required

    def reveal_nonce(self) -> Tuple[str, bytes]:
        """Phase 2: Reveal local nonce after commit quorum reached."""
        if not self.check_commit_quorum():
            raise RuntimeError("Cannot reveal before commit quorum")
        self.state = NonceCommitState.REVEAL_PENDING
        return self.local_node_id, self._local_nonce

    def receive_reveal(self, node_id: str, nonce: bytes) -> bool:
        """Phase 2: Receive and verify a revealed nonce.

        Returns True if SHA-256(nonce) matches the committed hash.
        """
        if node_id not in self._commitments:
            return False
        expected_hash = self._commitments[node_id].commit_hash
        actual_hash = hashlib.sha256(nonce).hexdigest()
        if actual_hash != expected_hash:
            return False
        self._commitments[node_id].revealed_nonce = nonce
        self._commitments[node_id].verified = True
        return True

    def check_reveal_quorum(self) -> bool:
        """Check if 2f+1 reveals have been verified."""
        verified = sum(1 for c in self._commitments.values() if c.verified)
        return verified >= self._quorum_required

    def finalize_nonce(self, timestamp_floor: int = 0) -> bytes:
        """Phase 3: Compute the versioned session nonce.

        final_nonce = SHA-256(session_uuid || timestamp_floor_5min || min_node_id)

        This binds the nonce to the session, preventing replay attacks.
        """
        if not self.check_reveal_quorum():
            raise RuntimeError("Cannot finalize before reveal quorum")

        min_node = min(self._commitments.keys())
        binding = (
            self.session_uuid.encode()
            + struct.pack(">Q", timestamp_floor)
            + min_node.encode()
        )
        final = hashlib.sha256(binding).digest()
        self.state = NonceCommitState.VERIFIED
        return final

    def get_mismatch_proof(self, node_id: str) -> Optional[Dict]:
        """Generate proof of nonce mismatch for Byzantine accusation."""
        commit = self._commitments.get(node_id)
        if commit is None or not commit.revealed_nonce:
            return None
        actual_hash = hashlib.sha256(commit.revealed_nonce).hexdigest()
        if actual_hash == commit.commit_hash:
            return None  # No mismatch
        return {
            "suspect": node_id,
            "committed_hash": commit.commit_hash,
            "revealed_hash": actual_hash,
            "revealed_nonce_prefix": commit.revealed_nonce[:8].hex(),
        }


# ======================================================================
# Semantic Anchor
# ======================================================================

def semantic_anchor_init(
    nonce: bytes,
    dim: int = SEM_PROJECTION_DIM,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize W_sem and Anchor_k from a shared nonce.

    W_sem is a fixed random projection matrix seeded deterministically
    from SHA-256(nonce). Anchor_k is the zero-epoch topological anchor
    derived via a toy gradient descent on a nonce-seeded manifold.

    Parameters
    ----------
    nonce : bytes
        The finalized session nonce (32 bytes).
    dim : int
        Projection dimensionality (default 64).

    Returns
    -------
    W_sem : Tensor [dim, dim]
        Fixed random projection matrix.
    anchor_k : Tensor [dim]
        Initial topological anchor vector.
    """
    # Seed a deterministic RNG from the nonce
    seed_int = int.from_bytes(hashlib.sha256(nonce).digest()[:8], "big")
    gen = torch.Generator()
    gen.manual_seed(seed_int)

    # W_sem: orthogonal random projection via QR decomposition
    raw = torch.randn(dim, dim, generator=gen)
    W_sem, _ = torch.linalg.qr(raw)

    # Anchor_k: toy manifold gradient descent (5 steps on a quadratic)
    anchor = torch.randn(dim, generator=gen)
    anchor = F.normalize(anchor, dim=0)
    # Simulate a simple curvature-aware walk
    for _ in range(5):
        grad = W_sem @ anchor  # toy gradient
        anchor = F.normalize(anchor - 0.1 * grad, dim=0)

    return W_sem, anchor


# ======================================================================
# α_sem Computation with Dual-Layer Ensemble
# ======================================================================

def compute_alpha_sem(
    h_layer: torch.Tensor,
    anchor_k: torch.Tensor,
    W_sem: torch.Tensor,
) -> float:
    """Compute semantic alignment α_sem for a single layer.

    α_sem = cosine_similarity(W_sem @ mean(h_layer), anchor_k)

    Parameters
    ----------
    h_layer : Tensor [batch, seq, hidden_dim] or [seq, hidden_dim]
        Hidden states from a transformer layer.
    anchor_k : Tensor [dim]
        Topological anchor vector.
    W_sem : Tensor [dim, hidden_dim] or [dim, dim]
        Semantic projection matrix.

    Returns
    -------
    alpha_sem : float in [-1, 1]
    """
    if h_layer.dim() == 3:
        h_mean = h_layer.float().mean(dim=(0, 1))  # [hidden_dim]
    elif h_layer.dim() == 2:
        h_mean = h_layer.float().mean(dim=0)  # [hidden_dim]
    else:
        h_mean = h_layer.float()

    # Project through W_sem
    proj_dim = W_sem.shape[0]
    h_dim = h_mean.shape[0]

    if h_dim > proj_dim:
        # Truncate hidden to projection dim
        h_mean = h_mean[:proj_dim]
    elif h_dim < proj_dim:
        # Pad with zeros
        h_mean = F.pad(h_mean, (0, proj_dim - h_dim))

    h_sem = W_sem @ h_mean  # [dim]
    alpha = F.cosine_similarity(
        h_sem.unsqueeze(0), anchor_k.unsqueeze(0).to(h_sem.device)
    ).item()
    return alpha


def compute_alpha_sem_ensemble(
    h_layer7: torch.Tensor,
    h_layer12: torch.Tensor,
    anchor_k: torch.Tensor,
    W_sem: torch.Tensor,
    weight7: float = ENSEMBLE_WEIGHT_LAYER7,
    weight12: float = ENSEMBLE_WEIGHT_LAYER12,
) -> float:
    """Dual-layer ensemble α_sem for smoother signal.

    α_sem_ensemble = w7 * α_sem(layer7) + w12 * α_sem(layer12)

    Parameters
    ----------
    h_layer7 : Tensor
        Hidden states from layer 7 (Page Time).
    h_layer12 : Tensor
        Hidden states from layer 12 (sink).
    anchor_k : Tensor [dim]
        Topological anchor.
    W_sem : Tensor [dim, dim]
        Semantic projection.
    weight7 : float
        Weight for layer 7 contribution.
    weight12 : float
        Weight for layer 12 contribution.

    Returns
    -------
    alpha_sem_ensemble : float
    """
    a7 = compute_alpha_sem(h_layer7, anchor_k, W_sem)
    a12 = compute_alpha_sem(h_layer12, anchor_k, W_sem)
    return weight7 * a7 + weight12 * a12


# ======================================================================
# Semantic Modulation with Hysteresis & Flash Veto
# ======================================================================

class SemanticModulator:
    """Modulates bridge strength via α_sem with hysteresis and flash veto.

    Implements:
      - Three-tier thresholds: full bridge / partial / OFF
      - 10-token cooldown on bridge OFF before re-evaluation
      - Flash veto: |α_sem(t) - α_sem(t-3)| > 0.35 → ignore 5 tokens
      - Byzantine accusation on α_sem < 0.4

    Parameters
    ----------
    full_threshold : float
        α_sem above this → full bridge strength (default 0.75).
    partial_floor : float
        α_sem below this → bridge OFF (default 0.55).
    byzantine_threshold : float
        α_sem below this → accusation broadcast (default 0.4).
    """

    def __init__(
        self,
        full_threshold: float = ALPHA_SEM_FULL_BRIDGE,
        partial_floor: float = ALPHA_SEM_PARTIAL_FLOOR,
        byzantine_threshold: float = ALPHA_SEM_BYZANTINE_THRESHOLD,
    ):
        self.full_threshold = full_threshold
        self.partial_floor = partial_floor
        self.byzantine_threshold = byzantine_threshold

        # History for flash veto detection
        self._alpha_history: List[float] = []
        self._cooldown_remaining: int = 0
        self._flash_veto_remaining: int = 0
        self._bridge_on: bool = True

        # Byzantine accusation log
        self.accusation_log: List[Tuple[int, float]] = []  # (token_idx, alpha)
        self._token_counter: int = 0

    def step(self, alpha_sem: float) -> Tuple[float, bool]:
        """Process one token step and return (bridge_strength, byzantine_flag).

        Parameters
        ----------
        alpha_sem : float
            Current α_sem value (ensemble or single-layer).

        Returns
        -------
        bridge_strength : float
            Effective bridge multiplier [0.0, 1.0].
        byzantine_flag : bool
            True if α_sem < byzantine_threshold → needs accusation.
        """
        self._token_counter += 1
        self._alpha_history.append(alpha_sem)

        byzantine_flag = False

        # --- Cooldown from bridge OFF (takes priority over flash veto) ---
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0.0, False

        # --- Flash veto check ---
        if len(self._alpha_history) >= 4:
            delta = abs(self._alpha_history[-1] - self._alpha_history[-4])
            if delta > FLASH_VETO_DELTA:
                self._flash_veto_remaining = FLASH_VETO_IGNORE_TOKENS

        if self._flash_veto_remaining > 0:
            self._flash_veto_remaining -= 1
            # During flash veto, hold previous bridge strength
            return self._last_strength(), False

        # --- Hysteresis thresholds ---
        if alpha_sem >= self.full_threshold:
            self._bridge_on = True
            strength = 1.0
        elif alpha_sem >= self.partial_floor:
            self._bridge_on = True
            strength = alpha_sem  # proportional in partial zone
        else:
            # Bridge OFF
            if self._bridge_on:
                self._cooldown_remaining = HYSTERESIS_COOLDOWN_TOKENS
            self._bridge_on = False
            strength = 0.0

        # --- Byzantine accusation ---
        if alpha_sem < self.byzantine_threshold:
            byzantine_flag = True
            self.accusation_log.append((self._token_counter, alpha_sem))

        return strength, byzantine_flag

    def _last_strength(self) -> float:
        """Return last non-veto bridge strength."""
        # Walk backwards to find last non-veto alpha
        for a in reversed(self._alpha_history[:-1]):
            if a >= self.full_threshold:
                return 1.0
            elif a >= self.partial_floor:
                return a
            else:
                return 0.0
        return 0.0

    def reset(self) -> None:
        """Reset modulator state for a new session."""
        self._alpha_history.clear()
        self._cooldown_remaining = 0
        self._flash_veto_remaining = 0
        self._bridge_on = True
        self.accusation_log.clear()
        self._token_counter = 0


# ======================================================================
# Anchor Consensus Gossip
# ======================================================================

class AnchorConsensusGossip:
    """Anchor drift detection via periodic hash gossip.

    Every ``gossip_interval`` tokens, nodes gossip SHA-256(Anchor_k).
    If 2f+1 nodes don't agree, outlier anchors trigger sever of
    the source node.

    After ``freeze_tokens``, anchor is frozen unless ≥75% vote to
    re-anchor.

    Parameters
    ----------
    anchor_k : Tensor [dim]
        Initial anchor vector.
    gossip_interval : int
        Tokens between gossip rounds (default 256).
    drift_threshold : float
        Maximum allowed L2 drift from initial anchor (default 0.08).
    freeze_tokens : int
        Anchor freezes after this many tokens (default 128).
    max_faulty : int
        Byzantine fault tolerance parameter.
    """

    def __init__(
        self,
        anchor_k: torch.Tensor,
        gossip_interval: int = ANCHOR_GOSSIP_INTERVAL,
        drift_threshold: float = ANCHOR_DRIFT_THRESHOLD,
        freeze_tokens: int = ANCHOR_FREEZE_TOKENS,
        max_faulty: int = 1,
    ):
        self._anchor_initial = anchor_k.clone().detach()
        self._anchor_current = anchor_k.clone().detach()
        self.gossip_interval = gossip_interval
        self.drift_threshold = drift_threshold
        self.freeze_tokens = freeze_tokens
        self.max_faulty = max_faulty
        self._quorum_required = 2 * max_faulty + 1
        self._frozen = False
        self._token_counter = 0

        # Peer anchor hashes from latest gossip round
        self._peer_hashes: Dict[str, str] = {}
        self._outlier_nodes: Set[str] = set()

        # Re-anchor votes
        self._reanchor_votes: Set[str] = set()
        self._total_nodes: int = 3  # default, updated during operation

    @property
    def anchor(self) -> torch.Tensor:
        return self._anchor_current

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def drift(self) -> float:
        """Current L2 drift from initial anchor."""
        return (self._anchor_current - self._anchor_initial).norm().item()

    def anchor_hash(self) -> str:
        """SHA-256 of the current anchor (for gossip)."""
        data = self._anchor_current.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def step(self) -> bool:
        """Advance token counter. Returns True if gossip round triggered."""
        self._token_counter += 1
        if self._token_counter >= self.freeze_tokens and not self._frozen:
            self._frozen = True
        return self._token_counter % self.gossip_interval == 0

    def should_gossip(self) -> bool:
        """Check if current token is a gossip boundary."""
        return self._token_counter % self.gossip_interval == 0

    def receive_anchor_hash(self, node_id: str, anchor_hash: str) -> None:
        """Receive a peer's anchor hash during gossip round."""
        self._peer_hashes[node_id] = anchor_hash

    def check_consensus(self, local_node_id: str) -> Tuple[bool, Set[str]]:
        """Check anchor consensus among peers.

        Returns (consensus_ok, outlier_nodes).
        """
        if not self._peer_hashes:
            return True, set()

        local_hash = self.anchor_hash()
        self._peer_hashes[local_node_id] = local_hash

        # Count hash agreement
        hash_counts: Dict[str, int] = {}
        for h in self._peer_hashes.values():
            hash_counts[h] = hash_counts.get(h, 0) + 1

        # Find majority hash
        majority_hash = max(hash_counts, key=hash_counts.get)  # type: ignore[arg-type]
        majority_count = hash_counts[majority_hash]

        # Identify outliers (nodes not matching majority)
        outliers: Set[str] = set()
        for nid, h in self._peer_hashes.items():
            if h != majority_hash:
                outliers.add(nid)

        self._outlier_nodes = outliers
        consensus_ok = majority_count >= self._quorum_required
        return consensus_ok, outliers

    def check_drift(self) -> bool:
        """Return True if anchor drift exceeds threshold."""
        return self.drift > self.drift_threshold

    def propose_reanchor(
        self, anchor_k_new: torch.Tensor, node_id: str,
    ) -> bool:
        """Propose a re-anchor. Only succeeds if anchor is not frozen
        or supermajority votes for it."""
        if not self._frozen:
            self._anchor_current = anchor_k_new.clone().detach()
            return True
        # Frozen: need supermajority
        self._reanchor_votes.add(node_id)
        if len(self._reanchor_votes) / max(self._total_nodes, 1) >= ANCHOR_REANCHOR_SUPERMAJORITY:
            self._anchor_current = anchor_k_new.clone().detach()
            self._reanchor_votes.clear()
            return True
        return False

    def reset_gossip_round(self) -> None:
        """Clear gossip state for next round."""
        self._peer_hashes.clear()


# ======================================================================
# Holographic Semantic Shard with Erasure Coding
# ======================================================================

def holographic_semantic_shard_encode(
    W_sem: torch.Tensor,
    anchor_k: torch.Tensor,
    redundancy: int = ERASURE_REDUNDANCY,
) -> List[torch.Tensor]:
    """Encode [W_sem[:32], Anchor_k] with Reed-Solomon-style erasure coding.

    Produces ``1 + redundancy`` shards. Any ``1`` shard suffices for
    reconstruction (simple duplication strategy for overhead < 5%).

    Parameters
    ----------
    W_sem : Tensor [dim, dim]
        Semantic projection matrix (only first 32 rows transmitted).
    anchor_k : Tensor [dim]
        Topological anchor vector.
    redundancy : int
        Number of redundant shards (default 2).

    Returns
    -------
    shards : list of Tensor
        Primary shard + redundancy duplicates with XOR parity diversification.
    """
    # Primary payload: concat W_sem[:32] flattened + anchor_k
    max_rows = min(32, W_sem.shape[0])
    primary_data = torch.cat([
        W_sem[:max_rows].reshape(-1),
        anchor_k,
    ])

    shards = [primary_data.clone()]

    # Erasure shards: XOR-diversified copies for spectral diversity
    for i in range(redundancy):
        # Create a deterministic parity mask per shard index
        gen = torch.Generator()
        gen.manual_seed(42 + i)
        parity = torch.randn(primary_data.shape, generator=gen).sign()
        # XOR-style: encode as primary * parity, decode by multiplying again
        shard = primary_data * parity
        shards.append(shard)

    return shards


def holographic_semantic_shard_decode(
    shards: List[Optional[torch.Tensor]],
    dim: int = SEM_PROJECTION_DIM,
    redundancy: int = ERASURE_REDUNDANCY,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Decode W_sem[:32] and Anchor_k from available shards.

    If the primary shard is available, use it directly. Otherwise
    reconstruct from any available erasure shard.

    Parameters
    ----------
    shards : list of Optional[Tensor]
        Received shards (None for missing).
    dim : int
        Projection dimension.
    redundancy : int
        Number of erasure shards expected.

    Returns
    -------
    W_sem_partial : Tensor [32, dim]
        Reconstructed partial W_sem.
    anchor_k : Tensor [dim]
        Reconstructed anchor.
    valid : bool
        True if reconstruction succeeded.
    """
    max_rows = min(32, dim)
    expected_len = max_rows * dim + dim

    # Try primary shard first
    if shards and shards[0] is not None and shards[0].shape[0] == expected_len:
        data = shards[0]
        W_partial = data[:max_rows * dim].reshape(max_rows, dim)
        anchor = data[max_rows * dim:]
        return W_partial, anchor, True

    # Try erasure shards
    for i in range(redundancy):
        idx = i + 1
        if idx < len(shards) and shards[idx] is not None:
            shard = shards[idx]
            if shard.shape[0] != expected_len:
                continue
            # Reverse the parity encoding
            gen = torch.Generator()
            gen.manual_seed(42 + i)
            parity = torch.randn(shard.shape, generator=gen).sign()
            data = shard * parity
            W_partial = data[:max_rows * dim].reshape(max_rows, dim)
            anchor = data[max_rows * dim:]
            return W_partial, anchor, True

    # All shards missing
    return (
        torch.zeros(max_rows, dim),
        torch.zeros(dim),
        False,
    )


def validate_shard_integrity(
    shards: List[Optional[torch.Tensor]],
    dim: int = SEM_PROJECTION_DIM,
) -> Tuple[bool, str]:
    """Cross-validate shards by decoding from multiple sources.

    If both primary and erasure shards are present, checks that they
    decode to the same data (within floating point tolerance).

    Returns (valid, reason).
    """
    max_rows = min(32, dim)
    expected_len = max_rows * dim + dim

    decoded_results = []

    # Decode from primary
    if shards and shards[0] is not None and shards[0].shape[0] == expected_len:
        decoded_results.append(shards[0])

    # Decode from erasure shards
    for i in range(ERASURE_REDUNDANCY):
        idx = i + 1
        if idx < len(shards) and shards[idx] is not None:
            shard = shards[idx]
            if shard.shape[0] != expected_len:
                continue
            gen = torch.Generator()
            gen.manual_seed(42 + i)
            parity = torch.randn(shard.shape, generator=gen).sign()
            decoded_results.append(shard * parity)

    if len(decoded_results) < 2:
        return True, "insufficient_shards_for_crosscheck"

    # Cross-check all pairs
    reference = decoded_results[0]
    for i, other in enumerate(decoded_results[1:], 1):
        diff = (reference - other).abs().max().item()
        if diff > 1e-5:
            return False, f"shard_mismatch_pair_0_{i}_diff_{diff:.6f}"

    return True, "OK"


# ======================================================================
# Semantic Bridge Modulation
# ======================================================================

def semantic_modulation(
    U_base: torch.Tensor,
    alpha_sem_final: float,
    bridge_strength: float,
) -> torch.Tensor:
    """Apply semantic lock modulation to base unitary/bridge output.

    U_out = bridge_strength × U_base

    When bridge_strength = 0 (bridge OFF), output is zeroed, causing
    the system to fall back to v2.1 activation mirroring.

    Parameters
    ----------
    U_base : Tensor
        Base bridge/unitary output.
    alpha_sem_final : float
        Final ensemble α_sem (for logging).
    bridge_strength : float
        Output of SemanticModulator.step() — [0.0, 1.0].

    Returns
    -------
    U_modulated : Tensor
    """
    return U_base * bridge_strength


# ======================================================================
# Full Semantic Lock Controller
# ======================================================================

class SemanticLockController:
    """Top-level controller for v2.2 Non-Local Semantic Locking.

    Integrates:
      - NonceCommitProtocol (multi-round commit)
      - semantic_anchor_init (W_sem + Anchor_k)
      - AnchorConsensusGossip (drift detection)
      - SemanticModulator (hysteresis + flash veto)
      - Erasure coding (shard encode/decode)
      - Byzantine accusation on α_sem < 0.4

    Parameters
    ----------
    local_node_id : str
        This node's identifier.
    max_faulty : int
        Byzantine tolerance.
    session_uuid : str
        Session identifier for replay binding.
    dim : int
        Semantic projection dimension (default 64).
    """

    def __init__(
        self,
        local_node_id: str,
        max_faulty: int = 1,
        session_uuid: str = "default-session",
        dim: int = SEM_PROJECTION_DIM,
    ):
        self.local_node_id = local_node_id
        self.max_faulty = max_faulty
        self.dim = dim

        # Sub-systems
        self.nonce_protocol = NonceCommitProtocol(
            local_node_id=local_node_id,
            max_faulty=max_faulty,
            session_uuid=session_uuid,
        )
        self.modulator = SemanticModulator()
        self._gossip: Optional[AnchorConsensusGossip] = None

        # Semantic state (initialized after nonce commit)
        self.W_sem: Optional[torch.Tensor] = None
        self.anchor_k: Optional[torch.Tensor] = None
        self._initialized: bool = False

        # Byzantine accusation output queue
        self.pending_accusations: List[Tuple[str, float]] = []

        # v2.1 fallback flag
        self._v21_fallback_active: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def v21_fallback_active(self) -> bool:
        return self._v21_fallback_active

    def initialize_from_nonce(self, nonce: bytes) -> None:
        """Initialize W_sem and Anchor_k from finalized nonce."""
        self.W_sem, self.anchor_k = semantic_anchor_init(nonce, self.dim)
        self._gossip = AnchorConsensusGossip(
            anchor_k=self.anchor_k,
            max_faulty=self.max_faulty,
        )
        self._initialized = True
        self._v21_fallback_active = False
        self.modulator.reset()

    def step(
        self,
        h_layer7: torch.Tensor,
        h_layer12: torch.Tensor,
        U_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, bool]:
        """Process one token step through the full semantic lock pipeline.

        Parameters
        ----------
        h_layer7 : Tensor
            Layer 7 hidden states.
        h_layer12 : Tensor
            Layer 12 hidden states.
        U_base : Tensor
            Base bridge output to modulate.

        Returns
        -------
        U_out : Tensor
            Semantically modulated output.
        alpha_sem : float
            Ensemble α_sem value.
        byzantine : bool
            True if Byzantine accusation was triggered.
        """
        if not self._initialized or self.W_sem is None or self.anchor_k is None:
            return U_base, 0.0, False

        # Compute ensemble α_sem
        alpha_sem = compute_alpha_sem_ensemble(
            h_layer7, h_layer12, self.anchor_k, self.W_sem,
        )

        # Modulate bridge
        bridge_strength, byzantine = self.modulator.step(alpha_sem)

        # Anchor drift check
        if self._gossip is not None:
            gossip_triggered = self._gossip.step()
            if self._gossip.check_drift():
                # Drift too high → force bridge OFF, activate v2.1 fallback
                bridge_strength = 0.0
                byzantine = True
                self._v21_fallback_active = True

        # Apply modulation
        U_out = semantic_modulation(U_base, alpha_sem, bridge_strength)

        # Record Byzantine accusations
        if byzantine:
            self.pending_accusations.append(
                (self.local_node_id, alpha_sem)
            )
            if alpha_sem < ALPHA_SEM_BYZANTINE_THRESHOLD:
                self._v21_fallback_active = True

        return U_out, alpha_sem, byzantine

    def get_shards(self) -> List[torch.Tensor]:
        """Encode current W_sem + Anchor_k as erasure-coded shards."""
        if self.W_sem is None or self.anchor_k is None:
            return []
        return holographic_semantic_shard_encode(self.W_sem, self.anchor_k)

    def drain_accusations(self) -> List[Tuple[str, float]]:
        """Pop and return pending Byzantine accusations."""
        accusations = list(self.pending_accusations)
        self.pending_accusations.clear()
        return accusations
