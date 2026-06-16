"""
handshake.py — Byzantine-Resistant Handshake (v2.0)
====================================================
Secure handshake protocol for dual-node entanglement with:

  - Precision class exchange and projector compatibility check.
  - Adaptive epoch length negotiation (conservative max).
  - Nonce exchange for signed message authentication.
  - Exception classes for incompatible nodes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .precision_projector import (
    PROJECTOR_REGISTRY,
    PrecisionClass,
    has_projector,
)


# ======================================================================
# Exception Classes
# ======================================================================

class IncompatibleNode(Exception):
    """Raised when a remote node cannot be paired due to precision
    or projector incompatibility."""


class HandshakeTimeout(Exception):
    """Raised when the remote node fails to respond within the
    expected timeframe."""


# ======================================================================
# Constants
# ======================================================================

PRECISION_CLASSES: Set[str] = {pc.value for pc in PrecisionClass}


# ======================================================================
# Synchronous Handshake (ZMQ-based)
# ======================================================================

def perform_handshake(
    pub_socket,
    sub_socket,
    local_node_id: str,
    local_precision: PrecisionClass,
    local_epoch_len: int = 16,
    timeout_ms: int = 5000,
    capability_proxy: Optional[float] = None,
) -> Dict[str, Any]:
    """Perform a synchronous Byzantine-resistant handshake over ZMQ.

    Parameters
    ----------
    pub_socket : zmq.Socket
        Publisher socket (already bound).
    sub_socket : zmq.Socket
        Subscriber socket (already connected).
    local_node_id : str
        This node's identifier.
    local_precision : PrecisionClass
        This node's quantization class.
    local_epoch_len : int
        Initial epoch length in tokens.
    timeout_ms : int
        Timeout in milliseconds for receiving the remote hello.
    capability_proxy : float, optional
        v2.1: Activation norm on a fixed prompt, used to populate
        RecursiveMirror.peer_capability for asymmetric kick scaling.

    Returns
    -------
    dict with keys: remote_id, remote_precision, epoch_len, nonce, my_nonce,
    remote_capability.

    Raises
    ------
    IncompatibleNode
        If precision or projector compatibility fails.
    HandshakeTimeout
        If the remote node doesn't respond in time.
    """
    import time
    import zmq

    supported = list(PROJECTOR_REGISTRY.keys())

    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)

    # Single-stage, slow-joiner robust exchange.
    #
    # Multi-stage PUB/SUB handshakes deadlock: once a node advances past the
    # hello stage it stops re-announcing, so a partner that has not yet
    # received that hello (its first send was dropped before the subscriber
    # connected) starves forever. We instead fold everything -- including this
    # node's nonce -- into ONE hello and re-publish it on a short interval until
    # the partner's hello arrives, then keep re-announcing for a brief grace
    # period so the partner is guaranteed to receive at least one of ours
    # before we exit. Both sides therefore converge within one grace window.
    my_nonce = os.urandom(32)
    hello = {
        "type": "HANDSHAKE",
        "node_id": local_node_id,
        "precision": local_precision.value,
        "epoch_len": local_epoch_len,
        "supported_projectors": [(s.value, t.value) for s, t in supported],
        "capability_proxy": capability_proxy,
        "nonce": my_nonce.hex(),
        # v2.3: Initial TPS estimate and clock offset for Chronos Lock
        "tps_estimate": 10.0,
        "clock_offset": 0.0,
    }

    deadline = time.monotonic() + timeout_ms / 1000.0
    resend_interval = 0.1   # re-announce every 100ms
    grace = 0.5             # keep announcing 500ms after first contact

    pub_socket.send_pyobj(hello)
    last_send = time.monotonic()
    remote_hello: Optional[Dict[str, Any]] = None
    first_contact = 0.0
    while True:
        now = time.monotonic()
        if remote_hello is None and now >= deadline:
            raise HandshakeTimeout(
                f"No handshake response within {timeout_ms}ms"
            )
        if remote_hello is not None and (now - first_contact) >= grace:
            break
        if now - last_send >= resend_interval:
            pub_socket.send_pyobj(hello)  # periodic resend vs slow-joiner
            last_send = now
        events = dict(poller.poll(int(resend_interval * 1000)))
        if sub_socket in events:
            m = sub_socket.recv_pyobj()
            if (
                isinstance(m, dict)
                and m.get("type") == "HANDSHAKE"
                and m.get("nonce")
                and remote_hello is None
            ):
                remote_hello = m
                first_contact = time.monotonic()

    # --- Validate precision ---
    remote_prec_str = remote_hello.get("precision")
    if remote_prec_str not in PRECISION_CLASSES:
        raise IncompatibleNode(f"Unknown precision: {remote_prec_str}")

    remote_precision = PrecisionClass(remote_prec_str)

    # --- Check projector availability (both directions) ---
    if not has_projector(local_precision, remote_precision):
        raise IncompatibleNode(
            f"No projector for {local_precision.value} -> {remote_precision.value}"
        )
    if not has_projector(remote_precision, local_precision):
        raise IncompatibleNode(
            f"No projector for {remote_precision.value} -> {local_precision.value}"
        )

    # --- Agree on epoch length (conservative max) ---
    remote_epoch = remote_hello.get("epoch_len", 16)
    agreed_epoch = max(local_epoch_len, remote_epoch)

    nonce = my_nonce
    remote_nonce_hex = remote_hello.get("nonce", "")

    return {
        "remote_id": remote_hello.get("node_id"),
        "remote_precision": remote_precision,
        "epoch_len": agreed_epoch,
        "nonce": bytes.fromhex(remote_nonce_hex),
        "my_nonce": nonce,
        "remote_capability": remote_hello.get("capability_proxy"),
        # v2.3: Chronos Lock initial estimates
        "remote_tps_estimate": remote_hello.get("tps_estimate", 10.0),
        "remote_clock_offset": remote_hello.get("clock_offset", 0.0),
    }


def validate_precision_pair(
    local: PrecisionClass, remote: PrecisionClass,
) -> bool:
    """Quick check: can these two precisions communicate (directly or via projector)?"""
    return has_projector(local, remote) and has_projector(remote, local)


def compute_capability_ratio(
    local_capability: Optional[float],
    remote_capability: Optional[float],
) -> float:
    """Compute capability ratio (remote / local) for VirtualLayer13.

    Returns 1.0 when either value is missing or zero.
    """
    if local_capability is None or remote_capability is None:
        return 1.0
    if local_capability <= 0:
        return 1.0
    return remote_capability / local_capability
