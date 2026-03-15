"""
dist/tier_manager.py — Tier admission and demotion for distributed nodes.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class NodeTier(Enum):
    COMPUTE = "COMPUTE"
    ROUTER = "ROUTER"


@dataclass
class _NodeRecord:
    node_id: str
    tier: NodeTier = NodeTier.ROUTER
    tps_ema: float = 0.0
    tps_variance: float = 0.0
    cumulative_wait: float = 0.0


class TierManager:
    """Manages tier assignment for distributed nodes."""

    def __init__(self, min_compute_tps: float = 10.0, max_wait_before_demotion: float = 2.0):
        self._min_compute_tps = min_compute_tps
        self._max_wait = max_wait_before_demotion
        self._records: dict[str, _NodeRecord] = {}

    def attest(self, node_id: str, tps_ema: float, tps_variance: float) -> NodeTier:
        tier = NodeTier.COMPUTE if tps_ema >= self._min_compute_tps else NodeTier.ROUTER
        self._records[node_id] = _NodeRecord(
            node_id=node_id, tier=tier,
            tps_ema=tps_ema, tps_variance=tps_variance,
        )
        return tier

    def record_wait(self, node_id: str, wait_secs: float) -> None:
        rec = self._records.get(node_id)
        if rec is None:
            return
        rec.cumulative_wait += wait_secs
        if rec.cumulative_wait > self._max_wait:
            rec.tier = NodeTier.ROUTER

    def get_record(self, node_id: str) -> _NodeRecord:
        return self._records[node_id]

    def compute_quorum_met(self, max_faulty: int) -> bool:
        n_compute = sum(1 for r in self._records.values() if r.tier == NodeTier.COMPUTE)
        return n_compute >= 2 * max_faulty + 1
