"""
test_chronos_lock.py — v2.3 Chronos Lock Hardening Tests
==========================================================
Covers:
  1. Cumulative desync: micro-lags (Δτ=0.19 for 8 tokens) → sever triggered.
  2. Sequence jump: hash validation and recovery request.
  3. TPS volatility: spike injection, EMA clipping and median fallback.
  4. RS coding: corrupt bytes in shard → graceful decode failure.
  5. Cayley vs Padé: entropy drift < 1e-6 on both paths.
  6. Probation: 3 waits >100ms → observer mode.
  7. Timestamp sync: offset estimation.
"""

from __future__ import annotations

import copy
import struct

import numpy as np
import pytest
import torch

from unitarity_labs.core.chronos_lock import (
    DESYNC_BASE_THRESHOLD,
    ENTROPY_DRIFT_LIMIT,
    PROBATION_CONSECUTIVE,
    PROBATION_TOKEN_PENALTY,
    TIMESTAMP_GOSSIP_INTERVAL,
    TPS_CLIP_MAX,
    TPS_CLIP_MIN,
    ChronosLock,
)
from unitarity_labs.core.kill_switch import ByzantineVoting, NodeStatus


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def lock():
    """Fresh ChronosLock for node A."""
    return ChronosLock(node_id="A")


@pytest.fixture
def lock_pair():
    """A pair of ChronosLock instances for chain validation tests."""
    return ChronosLock(node_id="A"), ChronosLock(node_id="B")


# ======================================================================
# 1. Cumulative Desync
# ======================================================================


class TestCumulativeDesync:
    def test_micro_lag_sever(self, lock):
        """8 × Δτ=0.19 yields integral ≈ 1.52 which exceeds threshold 0.2."""
        for _ in range(8):
            triggered = lock.update_desync(0.19)
        assert triggered, "Sever should trigger when integral_Δτ > threshold"

    def test_small_desync_no_sever(self, lock):
        """Very small Δτ should not trigger sever."""
        for _ in range(5):
            triggered = lock.update_desync(0.01)
        assert not triggered

    def test_integral_tracks_signed(self, lock):
        """Positive and negative Δτ should partially cancel."""
        lock.update_desync(0.15)
        lock.update_desync(0.15)
        lock.update_desync(-0.15)
        lock.update_desync(-0.15)
        assert abs(lock.integral_Δτ) < 0.01

    def test_adaptive_threshold_scales_with_nodes(self, lock):
        """Threshold grows (up to cap) with more nodes."""
        t2 = lock._adaptive_threshold(2)
        t8 = lock._adaptive_threshold(8)
        assert t8 > t2
        assert t8 <= 0.5

    def test_buffer_window_rolls(self, lock):
        """Verify only the last 32 values contribute."""
        for _ in range(40):
            lock.update_desync(0.005)
        assert len(lock.Δτ_buffer) == 32


# ======================================================================
# 2. Sequence Jump & Hash Chain
# ======================================================================


class TestSequenceChain:
    def test_hash_none_with_fewer_than_two_values(self, lock):
        """compute_τ_hash returns None if <2 τ values recorded."""
        assert lock.compute_τ_hash() is None
        lock.record_τ(1.0)
        assert lock.compute_τ_hash() is None

    def test_hash_deterministic(self, lock):
        """Same τ sequence → same hash."""
        lock.record_τ(1.0)
        lock.record_τ(2.0)
        h1 = lock.compute_τ_hash()
        lock2 = ChronosLock("B")
        lock2.record_τ(1.0)
        lock2.record_τ(2.0)
        h2 = lock2.compute_τ_hash()
        assert h1 == h2

    def test_validate_chain_initial(self, lock):
        """Initial step with None hash always passes."""
        assert lock.validate_τ_chain(None) is True

    def test_validate_chain_matching(self, lock_pair):
        """Matching hashes → valid chain."""
        a, b = lock_pair
        a.record_τ(1.0)
        a.record_τ(2.0)
        b.record_τ(1.0)
        b.record_τ(2.0)
        assert a.validate_τ_chain(b.prev_τ_hash) is True

    def test_validate_chain_mismatch(self, lock_pair):
        """Different τ sequences → chain validation fails."""
        a, b = lock_pair
        a.record_τ(1.0)
        a.record_τ(2.0)
        b.record_τ(1.0)
        b.record_τ(999.0)
        assert a.validate_τ_chain(b.prev_τ_hash) is False

    def test_handle_jump_normal(self, lock):
        """Jump of 1 is normal."""
        accept, request = lock.handle_jump(5, 6)
        assert accept is True
        assert request is False

    def test_handle_jump_small_recoverable(self, lock):
        """Jump of 2–5 triggers recovery request."""
        accept, request = lock.handle_jump(5, 8)
        assert accept is False
        assert request is True

    def test_handle_jump_severe(self, lock):
        """Jump > 5 rejects entirely."""
        accept, request = lock.handle_jump(5, 50)
        assert accept is False
        assert request is False


# ======================================================================
# 3. TPS Volatility
# ======================================================================


class TestTPSEstimation:
    def test_ema_basic(self, lock):
        """EMA tracks measurements."""
        for _ in range(50):
            lock.update_tps(50.0)
        assert abs(lock.tps_ema - 50.0) < 1.0

    def test_ema_clipping_low(self, lock):
        """TPS cannot fall below TPS_CLIP_MIN."""
        lock.update_tps(0.001)
        assert lock.tps_ema >= TPS_CLIP_MIN

    def test_ema_clipping_high(self, lock):
        """TPS cannot exceed TPS_CLIP_MAX."""
        lock.tps_ema = 195.0
        lock.update_tps(1000.0)
        assert lock.tps_ema <= TPS_CLIP_MAX

    def test_volatility_fallback_to_median(self, lock):
        """When std > 20, EMA switches to median."""
        # Fill history with alternating extreme values
        values = [1.0, 100.0, 2.0, 99.0, 3.0, 98.0, 1.0, 100.0, 2.0, 99.0]
        for v in values:
            lock.update_tps(v)
        # Median of those values should dominate
        median_val = float(np.median(values))
        assert abs(lock.tps_ema - median_val) < 5.0

    def test_no_volatility_fallback_when_stable(self, lock):
        """Stable TPS should not trigger median fallback."""
        # Pre-fill to avoid volatility interaction with initial tps_ema=10
        for _ in range(20):
            lock.update_tps(50.0)
        for v in [50.0, 51.0, 49.5, 50.2, 50.8, 49.9, 50.3, 51.1, 50.0, 49.7]:
            lock.update_tps(v)
        assert 45.0 < lock.tps_ema < 55.0


# ======================================================================
# 4. Reed-Solomon Shard Coding
# ======================================================================


class TestReedSolomonShard:
    def test_encode_decode_roundtrip(self, lock):
        """Clean shard encodes and decodes to original state."""
        lock.record_τ(1.5)
        lock.record_τ(2.5)
        lock.tps_ema = 42.0
        lock.update_desync(0.05)
        lock.update_desync(0.03)

        raw = lock.encode_shard()
        seq, τ, tps, prev_hash, integral = lock.decode_shard(raw)

        assert seq == lock.seq_pos
        assert abs(τ - 2.5) < 1e-5
        assert abs(tps - 42.0) < 1e-3
        assert prev_hash == lock.prev_τ_hash
        assert abs(integral - lock.integral_Δτ) < 1e-5

    def test_shard_size(self, lock):
        """Encoded shard is exactly 4 × 66 = 264 bytes."""
        lock.record_τ(1.0)
        raw = lock.encode_shard()
        assert len(raw) == 264

    def test_corrupt_shard_detected(self, lock):
        """Corrupting multiple bytes causes RS decode failure."""
        lock.record_τ(1.0)
        raw = bytearray(lock.encode_shard())
        # Corrupt many bytes in the first symbol block (exceeds RS correction)
        for i in range(0, 50):
            raw[i] ^= 0xFF
        with pytest.raises(ValueError, match="RS decoding failed"):
            lock.decode_shard(bytes(raw))

    def test_single_byte_correctable(self, lock):
        """Single byte error in a symbol block is correctable by RS."""
        lock.record_τ(3.14)
        lock.tps_ema = 25.0
        raw = bytearray(lock.encode_shard())
        # Corrupt 1 byte in the first block (RS(nsym=2) can correct 1 symbol error)
        raw[5] ^= 0xFF
        seq, τ, tps, prev_hash, integral = lock.decode_shard(bytes(raw))
        assert seq == lock.seq_pos

    def test_invalid_shard_size_rejected(self, lock):
        """Wrong-sized input is rejected immediately."""
        with pytest.raises(ValueError, match="Invalid shard size"):
            lock.decode_shard(b"\x00" * 100)


# ======================================================================
# 5. Unitary Wait Spin (Cayley vs Padé)
# ======================================================================


class TestUnitaryWaitSpin:
    def test_cayley_preserves_shape(self, lock):
        """Cayley wait spin preserves tensor shape."""
        h = torch.randn(2, 4, 16)
        h_out = lock.unitary_wait_spin(h, t_wait=0.001, mode="cayley")
        assert h_out.shape == h.shape

    def test_pade_preserves_shape(self, lock):
        """Padé wait spin preserves tensor shape."""
        h = torch.randn(2, 4, 16)
        h_out = lock.unitary_wait_spin(h, t_wait=0.001, mode="pade")
        assert h_out.shape == h.shape

    def test_zero_wait_is_identity(self, lock):
        """t_wait=0 returns input unchanged."""
        h = torch.randn(2, 4, 16)
        h_out = lock.unitary_wait_spin(h, t_wait=0.0)
        assert torch.equal(h, h_out)

    def test_cayley_entropy_drift_small(self, lock):
        """Cayley evolution has bounded entropy drift for small t."""
        h = torch.randn(4, 32)
        S_before = lock._entropy(h)
        h_out = lock.unitary_wait_spin(h, t_wait=1e-5, mode="cayley")
        S_after = lock._entropy(h_out)
        assert abs(S_after - S_before) <= ENTROPY_DRIFT_LIMIT

    def test_pade_entropy_drift_small(self, lock):
        """Padé evolution has bounded entropy drift for small t."""
        h = torch.randn(4, 32)
        S_before = lock._entropy(h)
        h_out = lock.unitary_wait_spin(h, t_wait=1e-5, mode="pade")
        S_after = lock._entropy(h_out)
        assert abs(S_after - S_before) <= ENTROPY_DRIFT_LIMIT

    def test_large_wait_cayley_may_raise(self, lock):
        """Very large t_wait with Cayley approx may trigger entropy drift abort."""
        h = torch.randn(4, 16)
        # Force the generator to be set
        lock.unitary_wait_spin(h, t_wait=1e-8, mode="cayley")
        # A very large t_wait will produce a non-unitary approximation
        with pytest.raises(RuntimeError, match="Entropy drift"):
            lock.unitary_wait_spin(h, t_wait=100.0, mode="cayley")


# ======================================================================
# 6. Probation
# ======================================================================


class TestProbation:
    def test_three_waits_triggers_observer(self, lock):
        """3 consecutive waits > 100ms → observer mode."""
        for _ in range(PROBATION_CONSECUTIVE):
            result = lock.check_probation(0.15)  # >100ms
        assert result is True

    def test_intermittent_waits_no_trigger(self, lock):
        """Alternating good/bad doesn't trigger probation."""
        for _ in range(10):
            lock.check_probation(0.15)
            lock.check_probation(0.01)
        assert lock.is_on_probation() is False

    def test_probation_sets_token_penalty(self, lock):
        """After probation, node is on probation for PROBATION_TOKEN_PENALTY tokens."""
        lock.seq_pos = 100
        for _ in range(PROBATION_CONSECUTIVE):
            lock.check_probation(0.15)
        assert lock.probation_until == 100 + PROBATION_TOKEN_PENALTY
        assert lock.is_on_probation() is True

    def test_probation_expires(self, lock):
        """Probation expires after enough tokens."""
        lock.seq_pos = 100
        for _ in range(PROBATION_CONSECUTIVE):
            lock.check_probation(0.15)
        lock.seq_pos = 100 + PROBATION_TOKEN_PENALTY + 1
        assert lock.is_on_probation() is False

    def test_consecutive_resets_after_probation(self, lock):
        """consecutive_waits resets to 0 after entering probation."""
        for _ in range(PROBATION_CONSECUTIVE):
            lock.check_probation(0.15)
        assert lock.consecutive_waits == 0


# ======================================================================
# 7. Timestamp Sync
# ======================================================================


class TestTimestampSync:
    def test_should_sync_after_interval(self, lock):
        """Sync is due after TIMESTAMP_GOSSIP_INTERVAL tokens."""
        lock.seq_pos = TIMESTAMP_GOSSIP_INTERVAL
        assert lock.should_timestamp_sync() is True

    def test_should_not_sync_early(self, lock):
        """Sync is not due before interval."""
        lock.seq_pos = 10
        assert lock.should_timestamp_sync() is False

    def test_prepare_timestamp_msg(self, lock):
        """Message has correct structure."""
        msg = lock.prepare_timestamp_msg()
        assert msg["type"] == "TIMESTAMP"
        assert msg["node"] == "A"
        assert isinstance(msg["ts"], float)

    def test_apply_timestamp_responses(self, lock):
        """Offset is computed as mean of (remote_ts - now) / 2."""
        import time

        now = time.monotonic()
        responses = {
            "B": {"ts": now + 0.01},
            "C": {"ts": now + 0.03},
        }
        offset = lock.apply_timestamp_responses(responses)
        # Offsets are small positive since remote ts > now
        assert isinstance(offset, float)
        assert lock.last_timestamp_sync == lock.seq_pos

    def test_apply_no_responses(self, lock):
        """No remote responses → offset stays at 0."""
        lock.apply_timestamp_responses({"A": {"ts": 1.0}})
        assert lock.system_clock_offset == 0.0


# ======================================================================
# 8. Kill-Switch Integration
# ======================================================================


class TestKillSwitchIntegration:
    def test_observer_status_exists(self):
        """NodeStatus.OBSERVER is available."""
        assert NodeStatus.OBSERVER == "OBSERVER"

    def test_observer_influence_nullified(self):
        """Observer nodes have nullified influence."""
        voting = ByzantineVoting(max_faulty=1)
        voting.set_observer("peer_X")
        assert voting.is_influence_nullified("peer_X") is True

    def test_desync_sever_sets_severed(self):
        """desync_sever immediately severs the node."""
        voting = ByzantineVoting(max_faulty=1)
        status = voting.desync_sever("peer_X", "local_A")
        assert status == NodeStatus.SEVERED

    def test_set_observer_only_from_active(self):
        """set_observer only demotes ACTIVE nodes."""
        voting = ByzantineVoting(max_faulty=1)
        voting.desync_sever("peer_X", "local_A")
        status = voting.set_observer("peer_X")
        assert status == NodeStatus.SEVERED  # stays SEVERED
