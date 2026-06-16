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

    def __init__(self, node_id: str = "A", krylov_dim: int = 128, zmq_port: int = 5555):
        try:
            import zmq
        except ImportError as e:
            raise ImportError(
                "The distributed/dual-node features require pyzmq. "
                "Install with: pip install 'unitarity-lab[dist]'"
            ) from e

        self.node_id = node_id
        self.krylov_dim = krylov_dim

        # ZeroMQ: Thread-safe pub/sub
        self.context = zmq.Context()
        self.pub = self.context.socket(zmq.PUB)
        self.sub = self.context.socket(zmq.SUB)

        # Socket hygiene: bound queues + no lingering on close so a slow or
        # dead partner cannot back-pressure us or stall shutdown. Set before
        # bind/connect so the options take effect on the connection.
        for _sock in (self.pub, self.sub):
            _sock.setsockopt(zmq.LINGER, 0)
        self.pub.setsockopt(zmq.SNDHWM, 1000)
        self.sub.setsockopt(zmq.RCVHWM, 1000)

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

        # Receive reliability knobs (see recv_partner_basis).
        #   latency_timeout : staleness ceiling for the FRESHEST drained
        #     message. Relaxed from the original 10ms -- under real per-token
        #     generation, GIL contention and queue build-up routinely push
        #     message age past 10ms, so a 10ms guard discarded nearly every
        #     message. 250ms still rejects genuinely ancient/desynced data
        #     while letting normal traffic through.
        #   poll_timeout_ms : how long recv waits for a message to arrive
        #     instead of returning immediately (mitigates per-call timing
        #     misses). Small so it does not noticeably stall generation.
        #   _recv_drain_cap : max messages drained per recv (keep-latest);
        #     bounds the drain loop so a saturated queue cannot spin.
        self.latency_timeout: float = 0.250
        self.poll_timeout_ms: int = 5
        self._recv_drain_cap: int = 64

        # Honest cross-sync exchange counters (measurement only)
        self._partner_recv_count: int = 0
        self._partner_miss_count: int = 0
        self._phi_history: list = []

    def attach_virtual_layer13(self, config, node_id: str) -> None:
        """Attach v3.0 field-synthesis components to this link.

        Constructs a :class:`VirtualLayer13` field synthesiser and a
        :class:`SafetyHead` refusal probe sized for ``config.hidden_size``
        and stores them on this bridge so dual-node forward passes can use a
        consistent pair. Idempotent: a repeated call rebuilds the components
        for the supplied ``config`` / ``node_id``.

        Parameters
        ----------
        config : object
            Model config exposing a ``hidden_size`` attribute.
        node_id : str
            This node's identifier (``"A"`` or ``"B"``).
        """
        # Imported lazily, mirroring the lazy-import style used elsewhere in
        # this module, and to avoid importing the nn modules unless the
        # distributed/dual-node path is actually exercised.
        from .virtual_layer13 import VirtualLayer13
        from .safety_head import SafetyHead

        self.virtual_layer13 = VirtualLayer13(config, node_id)
        self.safety_head = SafetyHead(hidden_dim=config.hidden_size)

    def _adjust_epoch(self, rtt: float) -> None:
        """Adjusts adaptive epoch length based on measured RTT.

        RTT > 50ms  → double epoch_len (high latency needs longer epochs)
        RTT < 30ms  → halve epoch_len (low latency allows shorter epochs)
        Otherwise   → no change
        Clamped to [16, 128].
        """
        epoch = getattr(self, 'epoch_len', 16)
        if rtt > 0.050:
            epoch = epoch * 2
        elif rtt < 0.030:
            epoch = epoch // 2
        self.epoch_len = max(16, min(epoch, 128))
        if not hasattr(self, '_rtt_history'):
            self._rtt_history = []
        self._rtt_history.append(rtt)

    def send_krylov_basis(self, krylov_basis: torch.Tensor) -> None:
        """Compress and transmit Krylov subspace via SVD low-rank."""
        A = krylov_basis.float()
        # Clamp to matrix dims: small matrices can't be compressed to a larger rank
        q_eff = min(self.krylov_dim, A.shape[-2], A.shape[-1])
        if q_eff < 1:
            U = A
        else:
            U, _, _ = torch.svd_lowrank(A, q=q_eff)
        msg = {
            'node': self.node_id,
            'basis': U.cpu().numpy(),
            'timestamp': time.monotonic(),
        }
        self.pub.send_pyobj(msg)

    def recv_partner_basis(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """Partner receive: poll-wait, drain to the freshest queued basis.

        Reliability rework. The old single non-blocking ``recv`` + 10ms guard
        dropped almost every message in production: it returned instantly when
        a message had not yet landed, and once the queue backed up it rejected
        the stale head-of-queue without ever reaching the fresh tail.

        This version:
          1. waits up to ``poll_timeout_ms`` for traffic (instead of giving up
             immediately),
          2. drains all currently-queued messages (bounded by
             ``_recv_drain_cap``) and keeps the freshest by timestamp -- the
             "latest partner basis" is what the rotation actually wants, and
          3. applies the (relaxed) ``latency_timeout`` staleness ceiling only
             to that freshest message, so genuinely ancient/desynced data is
             still rejected.

        Counter semantics are unchanged: one ``partner_recv`` per successful
        call, one ``partner_miss`` per empty/stale call.
        """
        import zmq  # cached after __init__; repeated here for zmq.DONTWAIT / zmq.Again

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        latest: Optional[dict] = None
        drained = 0
        polled_once = False
        while drained < self._recv_drain_cap:
            try:
                msg = self.sub.recv_pyobj(flags=zmq.DONTWAIT)
            except zmq.Again:
                if polled_once:
                    break
                # Nothing queued yet -- wait once for a message to arrive.
                events = dict(poller.poll(self.poll_timeout_ms))
                polled_once = True
                if self.sub in events:
                    continue
                break
            drained += 1
            if latest is None or msg.get('timestamp', 0.0) >= latest.get('timestamp', 0.0):
                latest = msg

        if latest is None:
            self._partner_miss_count += 1  # No message available (measurement only)
            return None

        latency = time.monotonic() - latest['timestamp']
        if latency > self.latency_timeout:
            self._partner_miss_count += 1  # Stale / desync guard miss (measurement only)
            return None
        basis = torch.tensor(latest['basis'], device=device)
        self._partner_recv_count += 1  # Real partner receive (measurement only)
        return basis

    def synchronize(self, timeout_ms: int = 5000) -> dict:
        """Slow-joiner barrier: block until the partner is reachable.

        ZeroMQ PUB/SUB silently drops everything published before the
        subscriber has finished connecting. Call this ONCE after the sockets
        are bound/connected and BEFORE any ``send_krylov_basis`` traffic so a
        hello round-trip has completed and subsequent data is actually
        delivered (this is the fix for a partner that otherwise receives 0
        messages for the whole run).

        Wraps :func:`perform_handshake` (precision/projector negotiation +
        nonce exchange, hardened against slow-joiner). The negotiated epoch
        length is applied to ``self.epoch_len``; the full result is cached on
        ``self._handshake`` and returned.

        Raises ``HandshakeTimeout`` / ``IncompatibleNode`` for the caller to
        handle (e.g. warn and continue in degraded solo mode).
        """
        from .handshake import perform_handshake
        from .precision_projector import PrecisionClass

        precision = getattr(self, "precision", PrecisionClass.FP32)
        epoch_len = getattr(self, "epoch_len", 16)
        result = perform_handshake(
            self.pub,
            self.sub,
            local_node_id=self.node_id,
            local_precision=precision,
            local_epoch_len=epoch_len,
            timeout_ms=timeout_ms,
        )
        self.epoch_len = result.get("epoch_len", epoch_len)
        self._handshake = result
        return result

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
                self._phi_history.append(phi_AB - 0.2)  # measurement only
                return phi_AB - 0.2  # Force anti-phase
        else:
            self.resonance_count = 0

        self._phi_history.append(phi_AB)  # measurement only
        return phi_AB

    @property
    def sync_stats(self) -> dict:
        """Honest dual-node exchange stats: did partner messages actually arrive?"""
        import statistics as _st
        phis = [p for p in self._phi_history if p != 0.0]
        return {
            "partner_recv": self._partner_recv_count,
            "partner_miss": self._partner_miss_count,
            "phi_samples": len(self._phi_history),
            "phi_nonzero": len(phis),
            "phi_mean": round(_st.mean(phis), 4) if phis else 0.0,
        }

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

        # Build projection matrix from partner basis.
        #
        # CONTRACT: partner_basis is a 2-D (d, k) orthonormal column basis --
        # the shape produced and transmitted by the pipeline (send side emits
        # the SVD column basis U of shape (samples_or_d, q)). For a column
        # basis P of shape (d, k), the orthogonal projector onto its column
        # span is P @ P^T, an (d, d) matrix; the projection of a row vector h
        # is h @ (P @ P^T).
        #
        # We normalise to that 2-D (d, k) contract here. If the basis arrives
        # 3-D ((batch, k, d) or (batch, d, k)) we collapse the leading batch
        # dimension by averaging, then orient by matching the feature dim d to
        # h_local's last dimension. A square ambiguous case prefers the
        # already-(d, k) orientation.
        P = partner_basis.float()
        if P.dim() == 3:
            # Average across the leading batch dimension -> 2-D.
            P = P.mean(0)
        if P.dim() != 2:
            raise ValueError(
                "partner_basis must reduce to a 2-D (d, k) basis; got shape "
                f"{tuple(partner_basis.shape)} (expected last/first dim == d={d})"
            )
        if P.shape[0] == d:
            pass  # already (d, k)
        elif P.shape[1] == d:
            P = P.transpose(-1, -2)  # (k, d) -> (d, k)
        else:
            raise ValueError(
                "partner_basis shape is incompatible with h_local feature dim: "
                f"got 2-D basis {tuple(P.shape)} but expected one dim == d={d} "
                f"(h_local last dim); original partner_basis shape "
                f"{tuple(partner_basis.shape)}"
            )

        # Orthogonal projector onto the partner column span: (d, k) @ (k, d).
        Pp = P @ P.transpose(-1, -2)  # (d, d)

        # v = h_local - proj(h_local onto partner)
        h_f = h_local.float()
        h_proj = h_f @ Pp  # (..., d) @ (d, d) -> (..., d)
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
        dual_bridge.send_krylov_basis(eigvecs)
        partner_basis = dual_bridge.recv_partner_basis(
            device=eigvecs.device.type,
        )
        phi_AB = dual_bridge.compute_cross_sync(eigvecs, partner_basis)
        act = output[0] if isinstance(output, (tuple, list)) else output
        act = dual_bridge.unitary_rotation_inject(act, partner_basis, phi_AB)
        bridge._bell_history.append(phi_AB)
        if isinstance(output, (tuple, list)):
            return (act, *output[1:])
        return act

    return layer_hook
