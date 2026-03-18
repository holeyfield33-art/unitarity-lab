"""
test_semantic_lock.py — v2.2 Semantic Lock Adversarial-Hardened Tests
======================================================================
Covers all five hardening vectors from Grok's adversarial audit:

  1. Nonce commitment protocol (multi-round, versioned binding)
  2. Anchor consensus gossip (drift detection, freeze, re-anchor)
  3. α_sem hysteresis + dual-layer ensemble + flash veto
  4. Holographic shard erasure coding (encode/decode/validate)
  5. Byzantine accusation + v2.1 fallback

Chain: Perplexity → Grok → DeepSeek → Copilot → Gemini
"""

from __future__ import annotations

import hashlib
import struct

import pytest
import torch
from typing import Optional, Sequence

from unitarity_labs.core.semantic_lock import (
    AnchorConsensusGossip,
    NonceCommitProtocol,
    NonceCommitState,
    SemanticLockController,
    SemanticModulator,
    compute_alpha_sem,
    compute_alpha_sem_ensemble,
    holographic_semantic_shard_decode,
    holographic_semantic_shard_encode,
    semantic_anchor_init,
    semantic_modulation,
    validate_shard_integrity,
    ALPHA_SEM_BYZANTINE_THRESHOLD,
    ALPHA_SEM_FULL_BRIDGE,
    ALPHA_SEM_PARTIAL_FLOOR,
    ANCHOR_DRIFT_THRESHOLD,
    ANCHOR_FREEZE_TOKENS,
    ANCHOR_GOSSIP_INTERVAL,
    FLASH_VETO_DELTA,
    SEM_PROJECTION_DIM,
)


# ======================================================================
# 1. Nonce Commitment Protocol
# ======================================================================

class TestNonceCommitProtocol:
    """Multi-round nonce hash-commit with versioned binding."""

    def test_generate_commit_produces_valid_hash(self):
        """Phase 1: commit hash should be valid SHA-256 hex."""
        proto = NonceCommitProtocol("node_A", max_faulty=1, session_uuid="s1")
        node_id, commit_hash = proto.generate_commit()
        assert node_id == "node_A"
        assert len(commit_hash) == 64
        assert all(c in "0123456789abcdef" for c in commit_hash)

    def test_state_transitions(self):
        """Protocol should progress through correct state machine."""
        p = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        assert p.state == NonceCommitState.IDLE
        p.generate_commit()
        assert p.state == NonceCommitState.COMMIT_SENT

    def test_commit_quorum_requires_2f_plus_1(self):
        """Quorum needs 2*1+1 = 3 commits for f=1."""
        p = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        p.generate_commit()  # A's own commit
        assert not p.check_commit_quorum()
        p.receive_commit("B", "b" * 64)
        assert not p.check_commit_quorum()
        p.receive_commit("C", "c" * 64)
        assert p.check_commit_quorum()

    def test_reveal_before_quorum_raises(self):
        """Cannot reveal nonce before commit quorum is reached."""
        p = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        p.generate_commit()
        with pytest.raises(RuntimeError, match="commit quorum"):
            p.reveal_nonce()

    def test_reveal_verify_honest(self):
        """Honest reveal verifies successfully."""
        p1 = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        p2 = NonceCommitProtocol("B", max_faulty=1, session_uuid="s1")
        p3 = NonceCommitProtocol("C", max_faulty=1, session_uuid="s1")

        # Phase 1: exchange commits
        _, h1 = p1.generate_commit()
        _, h2 = p2.generate_commit()
        _, h3 = p3.generate_commit()

        for p in [p1, p2, p3]:
            p.receive_commit("A", h1)
            p.receive_commit("B", h2)
            p.receive_commit("C", h3)

        # Phase 2: reveal and verify
        _, nonce_a = p1.reveal_nonce()
        assert p2.receive_reveal("A", nonce_a)
        assert p3.receive_reveal("A", nonce_a)

    def test_reveal_verify_byzantine_nonce_fails(self):
        """Byzantine node sending wrong nonce should fail verification."""
        p1 = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        p2 = NonceCommitProtocol("B", max_faulty=1, session_uuid="s1")
        p3 = NonceCommitProtocol("C", max_faulty=1, session_uuid="s1")

        _, h1 = p1.generate_commit()
        _, h2 = p2.generate_commit()
        _, h3 = p3.generate_commit()

        for p in [p1, p2, p3]:
            p.receive_commit("A", h1)
            p.receive_commit("B", h2)
            p.receive_commit("C", h3)

        # Byzantine: A sends a different nonce than committed
        fake_nonce = b"\x00" * 32
        assert not p2.receive_reveal("A", fake_nonce)

    def test_finalize_nonce_deterministic(self):
        """Same inputs → same finalized nonce (replay-resistant via binding)."""
        p = NonceCommitProtocol("A", max_faulty=0, session_uuid="s1")
        p.generate_commit()
        p._commitments["A"].verified = True
        n1 = p.finalize_nonce(timestamp_floor=1000)
        n2 = p.finalize_nonce(timestamp_floor=1000)
        assert n1 == n2

    def test_finalize_nonce_changes_with_session(self):
        """Different session UUIDs produce different nonces."""
        results = []
        for sid in ["session-1", "session-2"]:
            p = NonceCommitProtocol("A", max_faulty=0, session_uuid=sid)
            p.generate_commit()
            p._commitments["A"].verified = True
            results.append(p.finalize_nonce(timestamp_floor=0))
        assert results[0] != results[1]

    def test_mismatch_proof_on_honest_returns_none(self):
        """No mismatch proof when reveal is correct."""
        p = NonceCommitProtocol("A", max_faulty=0, session_uuid="s1")
        p.generate_commit()
        _, nonce = p.reveal_nonce()
        p.receive_reveal("A", nonce)
        assert p.get_mismatch_proof("A") is None

    def test_finalize_before_reveal_quorum_raises(self):
        """Cannot finalize before reveal quorum."""
        p = NonceCommitProtocol("A", max_faulty=1, session_uuid="s1")
        p.generate_commit()
        p.receive_commit("B", "b" * 64)
        p.receive_commit("C", "c" * 64)
        with pytest.raises(RuntimeError, match="reveal quorum"):
            p.finalize_nonce()


# ======================================================================
# 2. Semantic Anchor Initialization
# ======================================================================

class TestSemanticAnchorInit:
    """W_sem and Anchor_k from shared nonce."""

    def test_deterministic_from_nonce(self):
        """Same nonce → identical W_sem, anchor_k."""
        nonce = b"test_nonce_32_bytes_exactly_here!"
        W1, a1 = semantic_anchor_init(nonce, dim=64)
        W2, a2 = semantic_anchor_init(nonce, dim=64)
        assert torch.equal(W1, W2)
        assert torch.equal(a1, a2)

    def test_different_nonce_different_projection(self):
        """Different nonces → different geometry."""
        W1, a1 = semantic_anchor_init(b"nonce_A_padded_to_32bytes!!!!", dim=64)
        W2, a2 = semantic_anchor_init(b"nonce_B_padded_to_32bytes!!!!", dim=64)
        assert not torch.equal(W1, W2)
        assert not torch.equal(a1, a2)

    def test_W_sem_orthogonal(self):
        """W_sem should be orthogonal (QR decomposition)."""
        nonce = b"orthogonality_test_nonce_32byte!"
        W_sem, _ = semantic_anchor_init(nonce, dim=64)
        I = torch.eye(64)
        residual = (W_sem.T @ W_sem - I).norm().item()
        assert residual < 1e-5, f"W_sem not orthogonal: residual={residual}"

    def test_anchor_normalized(self):
        """Anchor_k should have unit norm."""
        nonce = b"normalization_test_nonce_32bytes"
        _, anchor = semantic_anchor_init(nonce, dim=64)
        assert abs(anchor.norm().item() - 1.0) < 1e-5

    def test_shapes(self):
        """Output shapes match requested dimension."""
        nonce = b"shape_test_nonce_exactly32bytes!"
        W, a = semantic_anchor_init(nonce, dim=32)
        assert W.shape == (32, 32)
        assert a.shape == (32,)


# ======================================================================
# 3. α_sem Computation
# ======================================================================

class TestAlphaSem:
    """Compute α_sem single-layer and ensemble."""

    @pytest.fixture
    def anchor_and_proj(self):
        nonce = b"alpha_sem_test_nonce_32bytes!!!!!"[:32]
        W_sem, anchor_k = semantic_anchor_init(nonce, dim=64)
        return W_sem, anchor_k

    def test_alpha_sem_bounded(self, anchor_and_proj):
        """α_sem should be in [-1, 1]."""
        W_sem, anchor_k = anchor_and_proj
        h = torch.randn(2, 16, 64)
        alpha = compute_alpha_sem(h, anchor_k, W_sem)
        assert -1.0 <= alpha <= 1.0

    def test_alpha_sem_2d_input(self, anchor_and_proj):
        """Should handle 2D input [seq, dim]."""
        W_sem, anchor_k = anchor_and_proj
        h = torch.randn(16, 64)
        alpha = compute_alpha_sem(h, anchor_k, W_sem)
        assert -1.0 <= alpha <= 1.0

    def test_ensemble_is_weighted_sum(self, anchor_and_proj):
        """Ensemble = 0.6 * layer7 + 0.4 * layer12."""
        W_sem, anchor_k = anchor_and_proj
        h7 = torch.randn(2, 16, 64)
        h12 = torch.randn(2, 16, 64)
        a7 = compute_alpha_sem(h7, anchor_k, W_sem)
        a12 = compute_alpha_sem(h12, anchor_k, W_sem)
        ensemble = compute_alpha_sem_ensemble(h7, h12, anchor_k, W_sem)
        expected = 0.6 * a7 + 0.4 * a12
        assert abs(ensemble - expected) < 1e-6

    def test_alpha_sem_handles_dim_mismatch(self, anchor_and_proj):
        """Should handle hidden_dim > projection_dim via truncation."""
        W_sem, anchor_k = anchor_and_proj
        h = torch.randn(2, 16, 128)  # hidden_dim > 64
        alpha = compute_alpha_sem(h, anchor_k, W_sem)
        assert -1.0 <= alpha <= 1.0


# ======================================================================
# 4. Semantic Modulator (Hysteresis + Flash Veto)
# ======================================================================

class TestSemanticModulator:
    """Three-tier hysteresis, cooldown, flash veto, Byzantine flag."""

    def test_full_bridge_above_threshold(self):
        """α_sem ≥ 0.75 → bridge_strength = 1.0."""
        m = SemanticModulator()
        strength, byz = m.step(0.80)
        assert strength == 1.0
        assert not byz

    def test_partial_bridge_in_middle_zone(self):
        """0.55 ≤ α_sem < 0.75 → bridge_strength = α_sem."""
        m = SemanticModulator()
        strength, byz = m.step(0.65)
        assert abs(strength - 0.65) < 1e-6
        assert not byz

    def test_bridge_off_below_floor(self):
        """α_sem < 0.55 → bridge OFF."""
        m = SemanticModulator()
        strength, byz = m.step(0.50)
        assert strength == 0.0

    def test_cooldown_after_bridge_off(self):
        """After bridge goes OFF, 10-token cooldown before re-eval."""
        m = SemanticModulator()
        # Start with bridge ON
        m.step(0.80)
        # Drop below floor → bridge OFF + cooldown
        m.step(0.40)
        # Next 10 tokens should return 0.0 regardless of α_sem
        for _ in range(10):
            strength, _ = m.step(0.90)
            assert strength == 0.0
        # After cooldown, should work normally
        strength, _ = m.step(0.90)
        assert strength == 1.0

    def test_flash_veto_transient_spike(self):
        """Large α_sem jump over 3 tokens triggers flash veto (5 tokens)."""
        m = SemanticModulator()
        # Establish baseline
        m.step(0.80)
        m.step(0.78)
        m.step(0.79)
        # Sudden drop > 0.35 from 3 tokens ago (0.80 → 0.40)
        strength, _ = m.step(0.40)
        # Flash veto should hold previous strength for 5 tokens
        for i in range(5):
            s, _ = m.step(0.80)
            # During flash veto, should hold last non-veto strength
            # (the value from before the spike, not the spike value)
            assert isinstance(s, float)
        # After veto expires, normal operation resumes
        s, _ = m.step(0.80)
        assert s == 1.0

    def test_byzantine_flag_below_0_4(self):
        """α_sem < 0.4 → byzantine_flag = True."""
        m = SemanticModulator()
        _, byz = m.step(0.35)
        assert byz

    def test_byzantine_flag_above_0_4(self):
        """α_sem ≥ 0.4 → byzantine_flag = False (unless bridge transition)."""
        m = SemanticModulator()
        _, byz = m.step(0.45)
        assert not byz

    def test_accusation_log_recorded(self):
        """Byzantine events should be logged with token index and α_sem."""
        m = SemanticModulator()
        m.step(0.9)
        m.step(0.3)  # should trigger
        assert len(m.accusation_log) == 1
        assert m.accusation_log[0][1] == 0.3

    def test_reset_clears_state(self):
        """Reset should clear all modulator state."""
        m = SemanticModulator()
        m.step(0.3)
        m.reset()
        assert len(m._alpha_history) == 0
        assert len(m.accusation_log) == 0
        assert m._cooldown_remaining == 0


# ======================================================================
# 5. Anchor Consensus Gossip
# ======================================================================

class TestAnchorConsensusGossip:
    """Anchor drift detection, freeze, re-anchor voting."""

    @pytest.fixture
    def gossip(self):
        anchor = torch.randn(64)
        anchor = anchor / anchor.norm()
        return AnchorConsensusGossip(
            anchor_k=anchor,
            gossip_interval=256,
            drift_threshold=0.08,
            freeze_tokens=128,
            max_faulty=1,
        )

    def test_initial_drift_zero(self, gossip):
        """Initial drift should be 0."""
        assert gossip.drift < 1e-7

    def test_anchor_hash_deterministic(self, gossip):
        """Same anchor → same hash."""
        h1 = gossip.anchor_hash()
        h2 = gossip.anchor_hash()
        assert h1 == h2

    def test_gossip_triggers_at_interval(self, gossip):
        """Gossip should trigger every 256 tokens."""
        for i in range(255):
            assert not gossip.step()
        assert gossip.step()  # token 256

    def test_anchor_freezes_after_threshold(self, gossip):
        """Anchor should freeze after freeze_tokens."""
        assert not gossip.frozen
        for _ in range(128):
            gossip.step()
        assert gossip.frozen

    def test_frozen_anchor_rejects_reanchor(self, gossip):
        """Frozen anchor rejects re-anchor without supermajority."""
        for _ in range(128):
            gossip.step()
        new_anchor = torch.randn(64)
        result = gossip.propose_reanchor(new_anchor, "node_A")
        assert not result  # only 1/3 nodes voted

    def test_reanchor_with_supermajority(self, gossip):
        """Supermajority (75%) allows re-anchor even when frozen."""
        gossip._total_nodes = 4
        for _ in range(128):
            gossip.step()
        new_anchor = torch.randn(64)
        gossip.propose_reanchor(new_anchor, "A")
        gossip.propose_reanchor(new_anchor, "B")
        result = gossip.propose_reanchor(new_anchor, "C")  # 3/4 = 75%
        assert result

    def test_consensus_detects_outlier(self, gossip):
        """Outlier node with different anchor hash is detected."""
        gossip.receive_anchor_hash("A", gossip.anchor_hash())
        gossip.receive_anchor_hash("B", gossip.anchor_hash())
        gossip.receive_anchor_hash("C", "badhash" + "0" * 56)
        ok, outliers = gossip.check_consensus("local")
        assert "C" in outliers

    def test_consensus_ok_with_agreement(self, gossip):
        """All nodes agreeing → consensus OK."""
        my_hash = gossip.anchor_hash()
        gossip.receive_anchor_hash("A", my_hash)
        gossip.receive_anchor_hash("B", my_hash)
        gossip.receive_anchor_hash("C", my_hash)
        ok, outliers = gossip.check_consensus("local")
        assert ok
        assert len(outliers) == 0

    def test_drift_detection(self, gossip):
        """Drift exceeding threshold is detected."""
        # Manually perturb anchor
        gossip._anchor_current = gossip._anchor_current + 0.1 * torch.ones(64)
        assert gossip.check_drift()

    def test_no_drift_within_threshold(self, gossip):
        """Small perturbation within threshold is OK."""
        gossip._anchor_current = gossip._anchor_current + 1e-4 * torch.ones(64)
        assert not gossip.check_drift()


# ======================================================================
# 6. Holographic Shard Erasure Coding
# ======================================================================

class TestHolographicShardErasure:
    """Encode/decode with erasure redundancy."""

    @pytest.fixture
    def sem_state(self):
        nonce = b"shard_test_nonce_exactly32bytes!"
        return semantic_anchor_init(nonce, dim=64)

    def test_encode_produces_correct_shard_count(self, sem_state):
        """1 primary + 2 redundancy = 3 shards."""
        W_sem, anchor_k = sem_state
        shards = holographic_semantic_shard_encode(W_sem, anchor_k, redundancy=2)
        assert len(shards) == 3

    def test_decode_from_primary_shard(self, sem_state):
        """Decode from primary shard should recover data."""
        W_sem, anchor_k = sem_state
        shards = holographic_semantic_shard_encode(W_sem, anchor_k)
        W_rec, a_rec, valid = holographic_semantic_shard_decode(shards)
        assert valid
        assert torch.allclose(W_rec, W_sem[:32], atol=1e-5)
        assert torch.allclose(a_rec, anchor_k, atol=1e-5)

    def test_decode_from_erasure_shard_only(self, sem_state):
        """Decode should work with only an erasure shard (primary missing)."""
        W_sem, anchor_k = sem_state
        encoded = holographic_semantic_shard_encode(W_sem, anchor_k)
        shards: list[Optional[torch.Tensor]] = [None] + encoded[1:]
        W_rec, a_rec, valid = holographic_semantic_shard_decode(shards)
        assert valid
        assert torch.allclose(W_rec, W_sem[:32], atol=1e-5)
        assert torch.allclose(a_rec, anchor_k, atol=1e-5)

    def test_decode_fails_all_missing(self, sem_state):
        """All shards missing → decode fails."""
        shards: Sequence[Optional[torch.Tensor]] = [None, None, None]
        _, _, valid = holographic_semantic_shard_decode(shards)
        assert not valid

    def test_shard_cross_validation_passes(self, sem_state):
        """Cross-validation of honest shards should pass."""
        W_sem, anchor_k = sem_state
        shards = holographic_semantic_shard_encode(W_sem, anchor_k)
        valid, reason = validate_shard_integrity(shards)
        assert valid
        assert reason == "OK"

    def test_shard_cross_validation_detects_tamper(self, sem_state):
        """Tampering a shard should fail cross-validation."""
        W_sem, anchor_k = sem_state
        encoded = holographic_semantic_shard_encode(W_sem, anchor_k)
        # Tamper primary shard
        shards: list[Optional[torch.Tensor]] = [encoded[0] + 1.0, encoded[1], encoded[2]]
        valid, reason = validate_shard_integrity(shards)
        assert not valid
        assert "mismatch" in reason

    def test_shard_shapes_consistent(self, sem_state):
        """All shards should have the same shape."""
        W_sem, anchor_k = sem_state
        shards = holographic_semantic_shard_encode(W_sem, anchor_k)
        shapes = {s.shape for s in shards}
        assert len(shapes) == 1

    def test_decode_second_erasure_shard(self, sem_state):
        """Decode from second erasure shard only."""
        W_sem, anchor_k = sem_state
        encoded = holographic_semantic_shard_encode(W_sem, anchor_k)
        # Remove primary and first erasure
        shards: list[Optional[torch.Tensor]] = [None, None] + encoded[2:]
        W_rec, a_rec, valid = holographic_semantic_shard_decode(shards)
        assert valid
        assert torch.allclose(a_rec, anchor_k, atol=1e-5)


# ======================================================================
# 7. Semantic Bridge Modulation
# ======================================================================

class TestSemanticModulation:
    """semantic_modulation applies bridge_strength scaling."""

    def test_full_strength_passthrough(self):
        """bridge_strength=1.0 → no change."""
        U = torch.randn(2, 16, 64)
        result = semantic_modulation(U, alpha_sem_final=0.8, bridge_strength=1.0)
        assert torch.equal(result, U)

    def test_zero_strength_zeros(self):
        """bridge_strength=0.0 → zero output (v2.1 fallback)."""
        U = torch.randn(2, 16, 64)
        result = semantic_modulation(U, alpha_sem_final=0.3, bridge_strength=0.0)
        assert (result == 0).all()

    def test_partial_strength_scales(self):
        """bridge_strength=0.6 → output scaled by 0.6."""
        U = torch.randn(2, 16, 64)
        result = semantic_modulation(U, alpha_sem_final=0.6, bridge_strength=0.6)
        assert torch.allclose(result, U * 0.6)


# ======================================================================
# 8. SemanticLockController (Integration)
# ======================================================================

class TestSemanticLockController:
    """Full pipeline: nonce → init → step → modulate → accuse."""

    @pytest.fixture
    def controller(self):
        ctrl = SemanticLockController(
            local_node_id="A",
            max_faulty=1,
            session_uuid="test-session",
            dim=64,
        )
        # Simulate completed nonce commit
        nonce = b"integration_test_nonce_32bytes!!"
        ctrl.initialize_from_nonce(nonce)
        return ctrl

    def test_initialized_after_nonce(self, controller):
        """Controller should be initialized after nonce commit."""
        assert controller.initialized
        assert controller.W_sem is not None
        assert controller.anchor_k is not None

    def test_not_initialized_before_nonce(self):
        """Controller should not be initialized before nonce commit."""
        ctrl = SemanticLockController("A", session_uuid="s")
        assert not ctrl.initialized

    def test_step_returns_modulated_output(self, controller):
        """Step should return modulated U_base."""
        h7 = torch.randn(2, 16, 64)
        h12 = torch.randn(2, 16, 64)
        U = torch.randn(2, 16, 64)
        U_out, alpha, byz = controller.step(h7, h12, U)
        assert U_out.shape == U.shape
        assert -1.0 <= alpha <= 1.0
        assert isinstance(byz, bool)

    def test_v21_fallback_on_low_alpha(self, controller):
        """Very low α_sem should trigger v2.1 fallback."""
        controller.modulator = SemanticModulator(
            full_threshold=0.99,  # Make it hard to get full bridge
            partial_floor=0.98,
            byzantine_threshold=10.0,  # Always trigger byzantine
        )
        h7 = torch.randn(2, 16, 64)
        h12 = torch.randn(2, 16, 64)
        U = torch.randn(2, 16, 64)
        _, _, byz = controller.step(h7, h12, U)
        assert byz
        assert controller.v21_fallback_active

    def test_get_shards_returns_erasure_coded(self, controller):
        """get_shards should return erasure-coded shard list."""
        shards = controller.get_shards()
        assert len(shards) == 3  # primary + 2 redundancy

    def test_drain_accusations_clears_queue(self, controller):
        """drain_accusations should pop and clear the queue."""
        controller.pending_accusations.append(("A", 0.3))
        acc = controller.drain_accusations()
        assert len(acc) == 1
        assert len(controller.pending_accusations) == 0

    def test_uninitialized_step_passthrough(self):
        """Uninitialized controller should pass U_base through."""
        ctrl = SemanticLockController("A", session_uuid="s")
        U = torch.randn(2, 16, 64)
        h = torch.randn(2, 16, 64)
        U_out, alpha, byz = ctrl.step(h, h, U)
        assert torch.equal(U_out, U)
        assert alpha == 0.0


# ======================================================================
# 9. RecursiveMirror Semantic Lock Integration
# ======================================================================

class TestRecursiveMirrorSemanticLock:
    """Ghost layer integration with SemanticLockController."""

    def test_attach_semantic_lock(self, toy_model):
        from unitarity_labs.core.ghost_layer import RecursiveMirror
        from unitarity_labs.core.bridge import CrossLayerEntanglementHook

        class MockCfg:
            hidden_size = 64
            num_attention_heads = 8

        bridge = CrossLayerEntanglementHook(
            toy_model, source_layer=7, sink_layer=12,
            coupling_strength=0.1, num_heads=8,
        )
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        mirror = RecursiveMirror(bridge=bridge, config=MockCfg())
        ctrl = SemanticLockController("A", session_uuid="s")
        ctrl.initialize_from_nonce(b"mirror_integration_nonce_32byte!")
        mirror.attach_semantic_lock(ctrl)
        assert mirror.semantic_lock is not None
        assert mirror.semantic_lock.initialized
        bridge.remove_hooks()

    def test_semantic_lock_none_by_default(self, toy_model):
        from unitarity_labs.core.ghost_layer import RecursiveMirror
        from unitarity_labs.core.bridge import CrossLayerEntanglementHook

        class MockCfg:
            hidden_size = 64
            num_attention_heads = 8

        bridge = CrossLayerEntanglementHook(
            toy_model, source_layer=7, sink_layer=12,
            coupling_strength=0.1, num_heads=8,
        )
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        mirror = RecursiveMirror(bridge=bridge, config=MockCfg())
        assert mirror.semantic_lock is None
        bridge.remove_hooks()


# ======================================================================
# 10. Adversarial Nonce Desync Attack
# ======================================================================

class TestAdversarialNonceDesync:
    """Nonce bit-flip attack vector from Grok audit."""

    def test_single_bit_flip_detection(self):
        """A 1-bit flip in the nonce should produce a different commit hash."""
        p1 = NonceCommitProtocol("A", max_faulty=0, session_uuid="s1")
        p1.generate_commit()
        nonce_a = p1._local_nonce

        # Flip one bit
        flipped = bytearray(nonce_a)
        flipped[0] ^= 0x01
        flipped = bytes(flipped)

        hash_original = hashlib.sha256(nonce_a).hexdigest()
        hash_flipped = hashlib.sha256(flipped).hexdigest()
        assert hash_original != hash_flipped

    def test_w_sem_diverges_on_bad_nonce(self):
        """Subtly different nonce → entirely different W_sem geometry."""
        nonce_good = b"good_nonce_padded_to_32bytes!!!!"
        nonce_bad = bytearray(nonce_good)
        nonce_bad[-1] ^= 0x01  # Single trailing bit flip
        nonce_bad = bytes(nonce_bad)

        W1, a1 = semantic_anchor_init(nonce_good, dim=64)
        W2, a2 = semantic_anchor_init(nonce_bad, dim=64)

        # Should be completely different
        cos_sim = torch.nn.functional.cosine_similarity(
            W1.reshape(1, -1), W2.reshape(1, -1)
        ).item()
        assert abs(cos_sim) < 0.5, f"W_sem too similar after bit flip: {cos_sim}"


# ======================================================================
# 11. Anchor Slow Drift Poisoning Attack
# ======================================================================

class TestAnchorSlowDriftPoisoning:
    """Gradual anchor drift below detection radar (Grok vector #2)."""

    def test_tight_threshold_catches_small_drift(self):
        """0.08 threshold should catch even modest drift."""
        anchor = torch.randn(64)
        anchor = anchor / anchor.norm()
        gossip = AnchorConsensusGossip(
            anchor_k=anchor, drift_threshold=0.08,
        )
        # Directed poisoning: consistent small shifts accumulate
        direction = torch.randn(64)
        direction = direction / direction.norm()
        for _ in range(100):
            gossip._anchor_current = gossip._anchor_current + 0.001 * direction
        assert gossip.check_drift(), "Tight threshold should catch gradual drift"

    def test_frozen_anchor_blocks_late_hijack(self):
        """After freeze, attacker can't slowly modify anchor."""
        anchor = torch.randn(64)
        anchor = anchor / anchor.norm()
        gossip = AnchorConsensusGossip(
            anchor_k=anchor, freeze_tokens=128,
        )
        for _ in range(128):
            gossip.step()
        assert gossip.frozen
        new_anchor = torch.randn(64)
        # Single node can't re-anchor
        assert not gossip.propose_reanchor(new_anchor, "attacker")


# ======================================================================
# 12. Flash Semantic Attack
# ======================================================================

class TestFlashSemanticAttack:
    """Transient entropy spike causing bridge OFF (Grok vector #3)."""

    def test_flash_veto_prevents_bridge_thrashing(self):
        """Rapid α_sem oscillation should be vetoed, not cause thrashing."""
        m = SemanticModulator()
        # Stable operation
        for _ in range(5):
            m.step(0.80)
        # Flash attack: sudden spike
        m.step(0.30)
        # Veto should prevent immediate bridge OFF from propagating
        # (holds previous strength during veto window)
        count_zero = 0
        for _ in range(5):
            s, _ = m.step(0.80)
            if s == 0.0:
                count_zero += 1
        # During veto, bridge should not be blindly OFF
        assert count_zero < 3, "Flash veto should prevent sustained zero-bridge"


# ======================================================================
# 13. Constants Validation
# ======================================================================

class TestConstants:
    """Validate hardened thresholds match Grok audit spec."""

    def test_anchor_drift_threshold(self):
        assert ANCHOR_DRIFT_THRESHOLD == 0.08

    def test_anchor_gossip_interval(self):
        assert ANCHOR_GOSSIP_INTERVAL == 256

    def test_anchor_freeze_tokens(self):
        assert ANCHOR_FREEZE_TOKENS == 128

    def test_alpha_sem_thresholds(self):
        assert ALPHA_SEM_FULL_BRIDGE == 0.75
        assert ALPHA_SEM_PARTIAL_FLOOR == 0.55
        assert ALPHA_SEM_BYZANTINE_THRESHOLD == 0.4

    def test_flash_veto_delta(self):
        assert FLASH_VETO_DELTA == 0.35

    def test_projection_dim(self):
        assert SEM_PROJECTION_DIM == 64
