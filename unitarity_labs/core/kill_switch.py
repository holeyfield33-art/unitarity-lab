"""
kill_switch.py — Byzantine Kill-Switch Hardening (v2.0)
========================================================
Leaderless Byzantine fault detection via β_TB trust metric:

  - β_TB < 0.20  → Hard Sever (immediate isolation + quorum ban).
  - 0.20 ≤ β_TB < 0.35  → Graceful Degradation (shadow gossip).
  - β_TB ≥ 0.45 for ≥ 5 epochs  → Re-admission.

Each node monitors β_TB locally. On violation it gossips a signed
``Suspect`` message. Upon receiving ``f+1`` distinct accusations the
node initiates a quorum vote requiring ``2f+1`` for global ban.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ======================================================================
# Constants / Thresholds
# ======================================================================

HARD_SEVER_THRESHOLD: float = 0.20
GRACEFUL_THRESHOLD: float = 0.35
READMIT_THRESHOLD: float = 0.45
READMIT_EPOCHS: int = 5
ACCUSATION_QUORUM_FACTOR: int = 1  # f+1 accusations to trigger vote


class NodeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SEVERED = "SEVERED"
    BANNED = "BANNED"
    OBSERVER = "OBSERVER"  # v2.3: Chronos probation


@dataclass
class NodeRecord:
    """Per-node trust tracking state."""
    node_id: str
    status: NodeStatus = NodeStatus.ACTIVE
    beta_history: List[float] = field(default_factory=list)
    consecutive_readmit_epochs: int = 0
    accusations: Set[str] = field(default_factory=set)
    ban_votes: Set[str] = field(default_factory=set)


# ======================================================================
# ByzantineVoting
# ======================================================================

class ByzantineVoting:
    """Leaderless Byzantine fault detector with kill-switch logic.

    Parameters
    ----------
    max_faulty : int
        Assumed maximum number of Byzantine nodes ``f``. Quorum
        requirements are derived from this value (2f+1 for ban,
        f+1 for accusation initiation).
    """

    def __init__(self, max_faulty: int = 1):
        self.f = max_faulty
        self._nodes: Dict[str, NodeRecord] = {}

    # ------------------------------------------------------------------
    # Node tracking
    # ------------------------------------------------------------------
    def _get_or_create(self, node_id: str) -> NodeRecord:
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeRecord(node_id=node_id)
        return self._nodes[node_id]

    def get_status(self, node_id: str) -> NodeStatus:
        return self._get_or_create(node_id).status

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def report_beta(self, node_id: str, beta: float) -> NodeStatus:
        """Report a β_TB observation for *node_id* and return new status.

        Decision logic:
          - β < HARD_SEVER  → SEVERED  (immediate isolation)
          - β < GRACEFUL    → DEGRADED (passive shadow gossip)
          - β ≥ READMIT for READMIT_EPOCHS consecutive → re-ACTIVE
          - otherwise       → status unchanged
        """
        rec = self._get_or_create(node_id)
        rec.beta_history.append(beta)

        # --- Hard Sever ---
        if beta < HARD_SEVER_THRESHOLD:
            rec.status = NodeStatus.SEVERED
            rec.consecutive_readmit_epochs = 0
            return rec.status

        # --- Graceful Degradation ---
        if beta < GRACEFUL_THRESHOLD:
            if rec.status not in (NodeStatus.SEVERED, NodeStatus.BANNED):
                rec.status = NodeStatus.DEGRADED
            rec.consecutive_readmit_epochs = 0
            return rec.status

        # --- Re-admission hysteresis ---
        if beta >= READMIT_THRESHOLD:
            rec.consecutive_readmit_epochs += 1
            if (
                rec.consecutive_readmit_epochs >= READMIT_EPOCHS
                and rec.status in (NodeStatus.DEGRADED, NodeStatus.SEVERED)
            ):
                rec.status = NodeStatus.ACTIVE
                rec.consecutive_readmit_epochs = 0
                rec.accusations.clear()
                rec.ban_votes.clear()
        else:
            rec.consecutive_readmit_epochs = 0

        return rec.status

    def suspect(self, suspect_id: str, accuser_id: str, reason: str = "") -> bool:
        """Record an accusation from *accuser_id* against *suspect_id*.

        Parameters
        ----------
        reason : str
            Optional human-readable reason (e.g. "Ψ_field hash mismatch").

        Returns True if the accusation count has reached the ``f+1``
        threshold needed to initiate a quorum vote.
        """
        rec = self._get_or_create(suspect_id)
        rec.accusations.add(accuser_id)
        if reason:
            if not hasattr(rec, "accusation_reasons"):
                rec.accusation_reasons: list[str] = []  # type: ignore[annotation-unchecked]
            rec.accusation_reasons.append(reason)  # type: ignore[attr-defined]
        return len(rec.accusations) >= self.f + ACCUSATION_QUORUM_FACTOR

    def cast_ban_vote(self, suspect_id: str, voter_id: str) -> None:
        """Cast a ban vote from *voter_id* against *suspect_id*."""
        rec = self._get_or_create(suspect_id)
        rec.ban_votes.add(voter_id)

    def quorum_check(self, suspect_id: str) -> bool:
        """Check if quorum (2f+1) has been reached for banning *suspect_id*.

        If quorum is reached, the node is moved to BANNED status.
        Returns True if the node is now banned.
        """
        rec = self._get_or_create(suspect_id)
        required = 2 * self.f + 1
        if len(rec.ban_votes) >= required:
            rec.status = NodeStatus.BANNED
            return True
        return False

    def is_influence_nullified(self, node_id: str) -> bool:
        """Nodes in DEGRADED/SEVERED/BANNED/OBSERVER have their influence
        on the collective phase nullified (starvation prevention)."""
        status = self.get_status(node_id)
        return status in (
            NodeStatus.DEGRADED, NodeStatus.SEVERED,
            NodeStatus.BANNED, NodeStatus.OBSERVER,
        )

    # ------------------------------------------------------------------
    # v2.3: Chronos desync sever + probation
    # ------------------------------------------------------------------

    def desync_sever(self, node_id: str, accuser_id: str) -> NodeStatus:
        """Sever a node due to cumulative desync (Chronos Lock).

        Immediately files accusation and sets SEVERED.
        """
        self.suspect(node_id, accuser_id)
        rec = self._get_or_create(node_id)
        rec.status = NodeStatus.SEVERED
        return rec.status

    def set_observer(self, node_id: str) -> NodeStatus:
        """Demote a node to observer mode (Chronos probation).

        Observer nodes receive shards but do not contribute.
        """
        rec = self._get_or_create(node_id)
        if rec.status == NodeStatus.ACTIVE:
            rec.status = NodeStatus.OBSERVER
        return rec.status

    # ------------------------------------------------------------------
    # v2.1: Quarantine integration (RecursiveMirror L_int > 0.8)
    # ------------------------------------------------------------------

    def quarantine_node(self, node_id: str, accuser_id: str) -> bool:
        """Handle a quarantine request from RecursiveMirror (L_int > 0.8).

        Files an accusation and, if quorum is reached, severs the node.

        Returns True if the node was banned as a result.
        """
        quorum_ready = self.suspect(node_id, accuser_id)
        if quorum_ready:
            self.cast_ban_vote(node_id, accuser_id)
            return self.quorum_check(node_id)
        # Even without quorum, degrade immediately on quarantine request
        rec = self._get_or_create(node_id)
        if rec.status == NodeStatus.ACTIVE:
            rec.status = NodeStatus.DEGRADED
        return False

    # ------------------------------------------------------------------
    # Integration with DualNodeEntanglementBridge
    # ------------------------------------------------------------------

    def evaluate_bridge_state(
        self,
        node_id: str,
        beta_tb: float,
        local_node_id: str,
    ) -> Tuple[NodeStatus, bool]:
        """Evaluate β_TB and decide on kill-switch action.

        Returns
        -------
        (new_status, should_sever)
            new_status : the updated NodeStatus for the remote node.
            should_sever : True if the link should be immediately severed.
        """
        status = self.report_beta(node_id, beta_tb)

        should_sever = False
        if status == NodeStatus.SEVERED:
            # Initiate accusation
            quorum_ready = self.suspect(node_id, local_node_id)
            should_sever = True

        return status, should_sever
