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
    import zmq

    supported = list(PROJECTOR_REGISTRY.keys())

    # --- Step 1: Send our hello ---
    hello = {
        "type": "HANDSHAKE",
        "node_id": local_node_id,
        "precision": local_precision.value,
        "epoch_len": local_epoch_len,
        "supported_projectors": [(s.value, t.value) for s, t in supported],
        "capability_proxy": capability_proxy,
        # v2.3: Initial TPS estimate and clock offset for Chronos Lock
        "tps_estimate": 10.0,
        "clock_offset": 0.0,
    }
    pub_socket.send_pyobj(hello)

    # --- Step 2: Receive remote hello ---
    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)
    events = dict(poller.poll(timeout_ms))

    if sub_socket not in events:
        raise HandshakeTimeout(
            f"No handshake response within {timeout_ms}ms"
        )
    remote_hello = sub_socket.recv_pyobj()

    # --- Step 3: Validate precision ---
    remote_prec_str = remote_hello.get("precision")
    if remote_prec_str not in PRECISION_CLASSES:
        raise IncompatibleNode(f"Unknown precision: {remote_prec_str}")

    remote_precision = PrecisionClass(remote_prec_str)

    # --- Step 4: Check projector availability (both directions) ---
    if not has_projector(local_precision, remote_precision):
        raise IncompatibleNode(
            f"No projector for {local_precision.value} -> {remote_precision.value}"
        )
    if not has_projector(remote_precision, local_precision):
        raise IncompatibleNode(
            f"No projector for {remote_precision.value} -> {local_precision.value}"
        )

    # --- Step 5: Agree on epoch length (conservative max) ---
    remote_epoch = remote_hello.get("epoch_len", 16)
    agreed_epoch = max(local_epoch_len, remote_epoch)

    # --- Step 6: Exchange nonces ---
    nonce = os.urandom(32)
    pub_socket.send_pyobj({"type": "NONCE", "nonce": nonce.hex()})

    events = dict(poller.poll(timeout_ms))
    if sub_socket not in events:
        raise HandshakeTimeout("No nonce response from remote node")
    nonce_msg = sub_socket.recv_pyobj()
    remote_nonce_hex = nonce_msg.get("nonce", "")

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
