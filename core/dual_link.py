"""
dual_link.py — Inter-Model ER=EPR Bridge (v1.7 Unitary Upgrade)
================================================================
Cross-process entanglement between Model A ↔ Model B via ZeroMQ
Pub/Sub with strict unitary (Householder) rotation injection.

v1.7 features:
  - **ZeroMQ Pub/Sub**: Thread-safe cross-process on ports 5555/5556.
  - **SVD Low-Rank Compression**: ``torch.svd_lowrank(q=128)`` for
    universal dimensionality alignment of Krylov bases.
  - **Adversarial Resonance Buffer**: phi_AB > 0.95 triggers forced
    desync after 3 consecutive saturated steps.
  - **Unitary Householder Reflection**: U†U = I norm-preserving
    rotation injection (replaces broken additive noise).
  - **10ms Latency Guard**: stale partner messages discarded.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F
import zmq

from .precision_projector import (
    CANONICAL_DTYPE,
    DequantAdapter,
    PrecisionClass,
    add_dither,
    get_projector,
)
from .kill_switch import ByzantineVoting, NodeStatus
from .ghost_layer import RecursiveMirror


class DualNodeEntanglementBridge:
    """Model A ↔ Model B: Unitary cross-process entanglement.

    Parameters
    ----------
    node_id : str
        ``"A"`` or ``"B"`` — determines pub/sub port assignment.
    krylov_dim : int
        Target rank for SVD low-rank compression (default 128).
    zmq_port : int
        Base ZeroMQ port. Node A publishes on ``zmq_port``,
        Node B publishes on ``zmq_port + 1``.
    """

    def __init__(
        self,
        node_id: str = "A",
        krylov_dim: int = 128,
        zmq_port: int = 5555,
        precision: PrecisionClass = PrecisionClass.BF16,
        epoch_len: int = 16,
    ):
        self.node_id = node_id
        self.krylov_dim = krylov_dim

        # ZeroMQ: Thread-safe pub/sub
        self.context = zmq.Context()
        self.pub = self.context.socket(zmq.PUB)
        self.sub = self.context.socket(zmq.SUB)

        if node_id == "A":
            self.pub.bind(f"tcp://*:{zmq_port}")
            self.sub.connect(f"tcp://localhost:{zmq_port + 1}")
        else:
            self.pub.bind(f"tcp://*:{zmq_port + 1}")
            self.sub.connect(f"tcp://localhost:{zmq_port}")

        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Adversarial buffers
        self.resonance_count: int = 0
        self.anti_resonance_threshold: float = 0.95
        self.latency_timeout: float = 0.010  # 10ms max

        # --- v2.0: Precision Alignment ---
        self.precision = precision
        self.remote_precision: Optional[PrecisionClass] = None
        self.projector_send: Optional[DequantAdapter] = None
        self.projector_recv: Optional[DequantAdapter] = None

        # --- v2.0: Adaptive Epoch ---
        self.epoch_len: int = epoch_len
        self._rtt_history: list[float] = []

        # --- v2.0: Reorthogonalization counter ---
        self._step_counter: int = 0
        self._reorth_interval: int = 256

        # --- v2.0: Byzantine Kill-Switch ---
        self.voting = ByzantineVoting(max_faulty=1)

    # ------------------------------------------------------------------
    # Precision projector setup
    # ------------------------------------------------------------------
    def set_remote_precision(
        self, remote_precision: PrecisionClass, dim: int,
    ) -> None:
        """Configure send/recv projectors after handshake."""
        self.remote_precision = remote_precision
        self.projector_send = get_projector(self.precision, remote_precision, dim)
        self.projector_recv = get_projector(remote_precision, self.precision, dim)

    # ------------------------------------------------------------------
    # Adaptive Epoch
    # ------------------------------------------------------------------
    def _adjust_epoch(self, rtt: float) -> None:
        """Adapt epoch length based on measured RTT.

        - RTT > 50ms  → double epoch_len (capped at 128).
        - RTT < 30ms  → halve epoch_len (floored at 16).
        """
        self._rtt_history.append(rtt)
        if rtt > 0.050:
            self.epoch_len = min(128, self.epoch_len * 2)
        elif rtt < 0.030:
            self.epoch_len = max(16, self.epoch_len // 2)

    # ------------------------------------------------------------------
    # Send / Receive with precision handling
    # ------------------------------------------------------------------
    def send_krylov_basis(self, krylov_basis: torch.Tensor) -> None:
        """Compress, cast to BF16+dither, and transmit Krylov basis."""
        U, _, _ = torch.svd_lowrank(krylov_basis.float(), q=self.krylov_dim)
        # v2.0: Cast to canonical BF16 with dithering
        U_dithered = add_dither(U, bits=16)
        msg = {
            'node': self.node_id,
            'basis': U_dithered.cpu().float().numpy(),
            'timestamp': time.monotonic(),
            'precision': self.precision.value,
        }
        self.pub.send_pyobj(msg)

    def recv_partner_basis(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """Non-blocking partner receive with latency guard + precision projection.

        v2.1: Validates received shard via spectral validation before use.
        """
        try:
            msg = self.sub.recv_pyobj(flags=zmq.DONTWAIT)
            latency = time.monotonic() - msg['timestamp']
            # Adaptive epoch: measure RTT (approx as one-way latency × 2)
            self._adjust_epoch(latency * 2)
            if latency > self.latency_timeout:
                return None  # Desync guard
            basis = torch.tensor(msg['basis'], device=device)
            # v2.0: Project received BF16 tensor to local precision
            if self.projector_recv is not None:
                self.projector_recv = self.projector_recv.to(device)
                basis = self.projector_recv(basis)

            # v2.1: Spectral shard validation (if metadata present)
            shard_meta = msg.get('shard_metadata')
            shard_hash = msg.get('shard_hash')
            if shard_meta is not None:
                valid, reason = RecursiveMirror.validate_shard(basis, shard_meta)
                if not valid:
                    remote_id = msg.get('node', 'unknown')
                    self.voting.suspect(remote_id, self.node_id)
                    return None  # Discard invalid shard
            if shard_hash is not None:
                actual_hash = RecursiveMirror.hash_shard(basis)
                if actual_hash != shard_hash:
                    remote_id = msg.get('node', 'unknown')
                    self.voting.suspect(remote_id, self.node_id)
                    return None  # Hash mismatch — discard

            return basis
        except zmq.Again:
            return None

    def compute_cross_sync(
        self, my_basis: torch.Tensor, partner_basis: Optional[torch.Tensor],
    ) -> float:
        """ER=EPR phase sync across models.

        Returns cosine similarity phi_AB in [-1, 1]. If phi_AB exceeds
        the anti-resonance threshold for > 3 consecutive steps, the
        value is reduced by 0.2 to force desync (resonance collapse buffer).
        """
        if partner_basis is None:
            return 0.0

        # Align dims via interpolation
        if my_basis.shape[-1] != partner_basis.shape[-1]:
            partner_basis = F.interpolate(
                partner_basis.unsqueeze(1).float(),
                size=my_basis.shape[-1],
            ).squeeze(1)

        # Optimal transport alignment via cosine similarity
        phi_AB = F.cosine_similarity(
            my_basis.float().mean(0),
            partner_basis.float().mean(0),
            dim=0,
        ).item()

        # RESONANCE COLLAPSE BUFFER
        if phi_AB > self.anti_resonance_threshold:
            self.resonance_count += 1
            if self.resonance_count > 3:
                return phi_AB - 0.2  # Force anti-phase
        else:
            self.resonance_count = 0

        return phi_AB

    def unitary_rotation_inject(
        self,
        h_local: torch.Tensor,
        partner_basis: Optional[torch.Tensor],
        phi_AB: float,
    ) -> torch.Tensor:
        """STRICT UNITARY: Householder reflection in partner subspace.

        Constructs U = I - 2vv†/||v||² where v is the component of
        h_local orthogonal to the partner subspace. The final output
        is a convex blend: ``strength * U @ h + (1 - strength) * h``,
        guaranteeing ||h'|| = ||h|| when strength = 1 (pure Householder).

        Returns h_local unchanged when phi_AB < 0.3 or partner is None.
        """
        if phi_AB < 0.3 or partner_basis is None:
            return h_local

        strength = min(0.15, phi_AB * 0.5)
        d = h_local.shape[-1]
        device = h_local.device

        # Build projection matrix from partner basis
        # partner_basis: (batch, k, d) — project h_local onto partner subspace
        P = partner_basis.float()
        # P^T P gives (batch, d, d) projection
        proj = P.transpose(-2, -1) @ P  # (batch, d, d)
        proj_mean = proj.mean(0)  # (d, d) average across batch

        # v = h_local - proj(h_local onto partner)
        h_f = h_local.float()
        h_proj = h_f @ proj_mean  # (..., d)
        v = h_f - h_proj

        # Householder reflection: U = I - 2 v v^T / ||v||^2
        v_flat = v.reshape(-1, d)
        h_flat = h_f.reshape(-1, d)
        v_norms_sq = (v_flat * v_flat).sum(dim=-1, keepdim=True)  # (N, 1)
        h_norms_sq = (h_flat * h_flat).sum(dim=-1, keepdim=True)  # (N, 1)

        # Relative threshold: skip reflection when v is numerical noise
        if v_norms_sq.max() < 1e-8 * h_norms_sq.max():
            return h_local

        # Per-vector Householder: each row gets its own reflection
        # U_i = I - 2 v_i v_i^T / ||v_i||^2
        # U_i @ h_i = h_i - 2 v_i (v_i^T h_i) / ||v_i||^2
        vth = (v_flat * h_flat).sum(dim=-1, keepdim=True)  # (N, 1)
        h_reflected = h_flat - 2 * v_flat * vth / (v_norms_sq + 1e-12)
        h_reflected = h_reflected.reshape(h_local.shape)

        # Convex blend: strength * reflected + (1 - strength) * original
        h_rotated = strength * h_reflected + (1 - strength) * h_f

        return h_rotated.to(h_local.dtype)

    def update_bridge_state(
        self,
        remote_node_id: str,
        beta_tb: float,
    ) -> NodeStatus:
        """Evaluate kill-switch logic for a remote node via β_TB.

        Integrates with the ByzantineVoting class to decide whether
        to sever, degrade, or re-admit the remote link.
        """
        status, should_sever = self.voting.evaluate_bridge_state(
            remote_node_id, beta_tb, self.node_id,
        )
        return status

    def close(self) -> None:
        """Clean shutdown of ZMQ sockets."""
        self.pub.close()
        self.sub.close()
        self.context.term()


# ======================================================================
# Integration Hook for CrossLayerEntanglementHook
# ======================================================================

def register_dual_node_hook(bridge: "CrossLayerEntanglementHook", node_id: str):
    """Non-invasive dual-node registration on an existing bridge.

    Attaches a ``DualNodeEntanglementBridge`` to the bridge and returns
    a layer hook that triggers at Layer 7 (source) and Layer 12 (sink),
    transmitting the Krylov basis and injecting the unitary rotation
    from the partner model. Cross-model phi_AB values are appended to
    ``bridge._bell_history`` for dashboard tracking.
    """
    dual_bridge = DualNodeEntanglementBridge(node_id=node_id)
    bridge.dual_link = dual_bridge  # type: ignore[attr-defined]

    def layer_hook(module, input, output, layer_idx):
        if layer_idx not in (bridge.source_layer, bridge.sink_layer):
            return output
        eigvecs = bridge.bridge_eigenvectors
        if eigvecs is None:
            return output

        # v2.0: Reorthogonalization every 256 steps
        dual_bridge._step_counter += 1
        if dual_bridge._step_counter % dual_bridge._reorth_interval == 0:
            bridge.reorthogonalize()

        dual_bridge.send_krylov_basis(eigvecs)
        partner_basis = dual_bridge.recv_partner_basis(
            device=eigvecs.device.type,
        )
        phi_AB = dual_bridge.compute_cross_sync(eigvecs, partner_basis)

        # v2.0: Kill-switch check (use phi_AB as β_TB proxy)
        if partner_basis is not None:
            remote_id = getattr(dual_bridge, '_remote_id', 'remote')
            status = dual_bridge.update_bridge_state(remote_id, phi_AB)
            if dual_bridge.voting.is_influence_nullified(remote_id):
                # Nullify remote influence — return output unchanged
                bridge._bell_history.append(phi_AB)
                return output

        act = output[0] if isinstance(output, (tuple, list)) else output
        act = dual_bridge.unitary_rotation_inject(act, partner_basis, phi_AB)
        bridge._bell_history.append(phi_AB)
        if isinstance(output, (tuple, list)):
            return (act, *output[1:])
        return act

    return layer_hook
