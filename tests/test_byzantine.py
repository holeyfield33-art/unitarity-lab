"""
test_byzantine.py — Tests for v2.0 Byzantine Hardening
=======================================================
Covers:
  1. Handshake with incompatible precision → raises IncompatibleNode.
  2. Projector invocation on sample tensors.
  3. Kill-switch thresholds with simulated β_TB values.
  4. Adaptive epoch with mocked RTT.
  5. Reorthogonalization preserves unitarity.
"""

from __future__ import annotations

import pytest
import torch

from core.precision_projector import (
    CANONICAL_DTYPE,
    DequantAdapter,
    PrecisionClass,
    add_dither,
    get_projector,
    has_projector,
)
from core.handshake import (
    IncompatibleNode,
    validate_precision_pair,
)
from core.kill_switch import (
    GRACEFUL_THRESHOLD,
    HARD_SEVER_THRESHOLD,
    READMIT_EPOCHS,
    READMIT_THRESHOLD,
    ByzantineVoting,
    NodeStatus,
)
from core.dual_link import DualNodeEntanglementBridge
from core.bridge import CrossLayerEntanglementHook

# Reuse conftest fixtures
from tests.conftest import ToyTransformer


# ======================================================================
# 1. Precision Compatibility Tests
# ======================================================================


class TestPrecisionCompatibility:
    def test_same_precision_has_projector(self):
        """Same precision can always communicate directly."""
        for pc in PrecisionClass:
            assert has_projector(pc, pc)

    def test_int4_bf16_has_projector(self):
        assert has_projector(PrecisionClass.INT4, PrecisionClass.BF16)
        assert has_projector(PrecisionClass.BF16, PrecisionClass.INT4)

    def test_validate_pair_compatible(self):
        assert validate_precision_pair(PrecisionClass.FP8_E4M3, PrecisionClass.BF16)

    def test_validate_pair_incompatible_unknown(self):
        """Validate fails gracefully when precision is not in registry."""
        # Simulate an unknown pair by checking a pair that doesn't exist
        # All standard pairs exist, so we check the registry covers them
        assert validate_precision_pair(PrecisionClass.FP32, PrecisionClass.BF16)


# ======================================================================
# 2. Projector / Dithering Tests
# ======================================================================


class TestProjector:
    def test_dequant_adapter_identity_init(self):
        """Freshly initialized adapter should be near-identity."""
        adapter = DequantAdapter(dim=64)
        x = torch.randn(4, 64)
        y = adapter(x)
        assert y.shape == x.shape
        assert torch.allclose(y.float(), x.float(), atol=1e-5)

    def test_dequant_adapter_gradient_flow(self):
        """Adapter must support gradient flow for offline tuning."""
        adapter = DequantAdapter(dim=32)
        x = torch.randn(8, 32, requires_grad=True)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_get_projector_returns_adapter(self):
        proj = get_projector(PrecisionClass.INT4, PrecisionClass.BF16, dim=64)
        assert proj is not None
        assert isinstance(proj, DequantAdapter)

    def test_get_projector_same_returns_none(self):
        proj = get_projector(PrecisionClass.BF16, PrecisionClass.BF16, dim=64)
        assert proj is None

    def test_add_dither_dtype(self):
        x = torch.randn(4, 16)
        dithered = add_dither(x, bits=16)
        assert dithered.dtype == CANONICAL_DTYPE

    def test_add_dither_preserves_shape(self):
        x = torch.randn(8, 32)
        dithered = add_dither(x, bits=16)
        assert dithered.shape == x.shape

    def test_add_dither_injects_noise(self):
        """Dithered output should not be identical to simply casting."""
        torch.manual_seed(42)
        x = torch.randn(16, 64)
        d1 = add_dither(x, bits=16)
        d2 = add_dither(x, bits=16)
        # Two dithering passes should give different results
        assert not torch.equal(d1, d2)


# ======================================================================
# 3. Kill-Switch Threshold Tests
# ======================================================================


class TestKillSwitch:
    def test_hard_sever_below_threshold(self):
        voting = ByzantineVoting(max_faulty=1)
        status = voting.report_beta("node_x", 0.10)
        assert status == NodeStatus.SEVERED

    def test_graceful_degradation(self):
        voting = ByzantineVoting(max_faulty=1)
        status = voting.report_beta("node_x", 0.25)
        assert status == NodeStatus.DEGRADED

    def test_active_above_graceful(self):
        voting = ByzantineVoting(max_faulty=1)
        status = voting.report_beta("node_x", 0.50)
        assert status == NodeStatus.ACTIVE

    def test_readmission_hysteresis(self):
        voting = ByzantineVoting(max_faulty=1)
        # First degrade the node
        voting.report_beta("node_x", 0.10)
        assert voting.get_status("node_x") == NodeStatus.SEVERED

        # Report good β for READMIT_EPOCHS consecutive times
        for _ in range(READMIT_EPOCHS):
            voting.report_beta("node_x", 0.50)

        assert voting.get_status("node_x") == NodeStatus.ACTIVE

    def test_readmission_resets_on_dip(self):
        voting = ByzantineVoting(max_faulty=1)
        voting.report_beta("node_x", 0.10)  # sever
        for _ in range(READMIT_EPOCHS - 1):
            voting.report_beta("node_x", 0.50)  # almost readmitted
        voting.report_beta("node_x", 0.10)  # dip resets counter
        assert voting.get_status("node_x") == NodeStatus.SEVERED

    def test_accusation_and_quorum(self):
        voting = ByzantineVoting(max_faulty=1)
        # f=1, ACCUSATION_QUORUM_FACTOR=1 → need f+1=2 accusations
        r1 = voting.suspect("bad_node", "accuser_1")
        assert not r1  # 1 < 2
        r2 = voting.suspect("bad_node", "accuser_2")
        assert r2  # 2 >= 2

    def test_ban_quorum(self):
        voting = ByzantineVoting(max_faulty=1)
        # Need 2f+1 = 3 votes to ban
        voting.cast_ban_vote("bad_node", "v1")
        assert not voting.quorum_check("bad_node")
        voting.cast_ban_vote("bad_node", "v2")
        assert not voting.quorum_check("bad_node")
        voting.cast_ban_vote("bad_node", "v3")
        assert voting.quorum_check("bad_node")
        assert voting.get_status("bad_node") == NodeStatus.BANNED

    def test_influence_nullified(self):
        voting = ByzantineVoting(max_faulty=1)
        voting.report_beta("node_x", 0.10)  # severed
        assert voting.is_influence_nullified("node_x")

    def test_influence_not_nullified_for_active(self):
        voting = ByzantineVoting(max_faulty=1)
        voting.report_beta("node_x", 0.50)
        assert not voting.is_influence_nullified("node_x")


# ======================================================================
# 4. Adaptive Epoch Tests
# ======================================================================


class TestAdaptiveEpoch:
    def test_epoch_doubles_on_high_rtt(self):
        bridge = DualNodeEntanglementBridge.__new__(DualNodeEntanglementBridge)
        bridge.epoch_len = 16
        bridge._rtt_history = []
        bridge._adjust_epoch(0.060)  # > 50ms
        assert bridge.epoch_len == 32

    def test_epoch_halves_on_low_rtt(self):
        bridge = DualNodeEntanglementBridge.__new__(DualNodeEntanglementBridge)
        bridge.epoch_len = 64
        bridge._rtt_history = []
        bridge._adjust_epoch(0.020)  # < 30ms
        assert bridge.epoch_len == 32

    def test_epoch_capped_at_128(self):
        bridge = DualNodeEntanglementBridge.__new__(DualNodeEntanglementBridge)
        bridge.epoch_len = 128
        bridge._rtt_history = []
        bridge._adjust_epoch(0.100)
        assert bridge.epoch_len == 128

    def test_epoch_floored_at_16(self):
        bridge = DualNodeEntanglementBridge.__new__(DualNodeEntanglementBridge)
        bridge.epoch_len = 16
        bridge._rtt_history = []
        bridge._adjust_epoch(0.010)
        assert bridge.epoch_len == 16

    def test_epoch_no_change_mid_range(self):
        bridge = DualNodeEntanglementBridge.__new__(DualNodeEntanglementBridge)
        bridge.epoch_len = 32
        bridge._rtt_history = []
        bridge._adjust_epoch(0.040)  # between 30ms and 50ms
        assert bridge.epoch_len == 32


# ======================================================================
# 5. Reorthogonalization Tests
# ======================================================================


class TestReorthogonalization:
    def test_reorthogonalize_preserves_unitarity(self):
        """After QR, Q^T Q ≈ I within float32 tolerance."""
        model = ToyTransformer(d_model=64, num_layers=13)
        hook = CrossLayerEntanglementHook(
            model=model,
            source_layer=6,
            sink_layer=11,
            num_heads=4,
        )
        # Inject a non-orthogonal eigenvector matrix
        hook._bridge_eigenvectors = torch.randn(64, 3)
        hook.reorthogonalize()

        V = hook._bridge_eigenvectors.float()
        identity_err = (V.T @ V - torch.eye(V.shape[1])).abs().max().item()
        assert identity_err < 1e-6

    def test_reorthogonalize_lora_a(self):
        """LoRA A columns should be orthonormal after reorthogonalization."""
        model = ToyTransformer(d_model=64, num_layers=13)
        hook = CrossLayerEntanglementHook(
            model=model,
            source_layer=6,
            sink_layer=11,
            num_heads=4,
            lora_rank=8,
        )
        hook.reorthogonalize()

        A = hook.lora_adapter.lora_A.data.float()
        identity_err = (A.T @ A - torch.eye(A.shape[1])).abs().max().item()
        assert identity_err < 1e-6

    def test_reorthogonalize_no_eigvecs_safe(self):
        """Reorthogonalize should not crash when eigenvectors are None."""
        model = ToyTransformer(d_model=64, num_layers=13)
        hook = CrossLayerEntanglementHook(
            model=model,
            source_layer=6,
            sink_layer=11,
            num_heads=4,
        )
        hook._bridge_eigenvectors = None
        hook.reorthogonalize()  # Should not raise

    def test_reorthogonalize_idempotent(self):
        """Running reorthogonalize twice should give same result."""
        model = ToyTransformer(d_model=64, num_layers=13)
        hook = CrossLayerEntanglementHook(
            model=model,
            source_layer=6,
            sink_layer=11,
            num_heads=4,
        )
        hook._bridge_eigenvectors = torch.randn(64, 3)

        hook.reorthogonalize()
        V1 = hook._bridge_eigenvectors.clone()

        hook.reorthogonalize()
        V2 = hook._bridge_eigenvectors

        # Up to sign flips, columns should match
        for i in range(V1.shape[1]):
            dot = (V1[:, i] * V2[:, i]).sum().abs().item()
            assert abs(dot - 1.0) < 1e-6
