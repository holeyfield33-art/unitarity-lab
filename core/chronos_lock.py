"""
core/chronos_lock.py — Temporal Synchronization (v3.0.0-Singularity)
=====================================================================
Distributed-only subsystem. Not required for single-node operation.

Provides:
- Causal entrainment via velocity-normalized τ_sync
- Unitary idle spins (Padé / Cayley) with entropy monitoring
- Cumulative desync tracking and adaptive sever
- Sequence continuity hashing and jump recovery
- Robust TPS estimation (clipped EMA + volatility median)
- Reed-Solomon encoded temporal shards
- Periodic timestamp gossip for clock drift correction
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
from reedsolo import RSCodec, ReedSolomonError


# ======================================================================
# Constants
# ======================================================================

TPS_CLIP_MIN: float = 0.5
TPS_CLIP_MAX: float = 200.0
TPS_VOLATILITY_THRESHOLD: float = 20.0
TPS_HISTORY_LEN: int = 10

DESYNC_WINDOW: int = 32
DESYNC_BASE_THRESHOLD: float = 0.2
DESYNC_MAX_THRESHOLD: float = 0.5

PROBATION_WAIT_THRESHOLD: float = 0.1   # 100ms
PROBATION_CONSECUTIVE: int = 3
PROBATION_TOKEN_PENALTY: int = 128

SEQUENCE_MAX_RECOVERABLE_JUMP: int = 5

RS_NSYM: int = 2          # parity symbols per block
SHARD_DATA_LEN: int = 256  # 4 symbols × 64 bytes
SHARD_SYMBOL_LEN: int = 64

TIMESTAMP_GOSSIP_INTERVAL: int = 1024  # tokens

ENTROPY_DRIFT_LIMIT: float = 1e-6


class ChronosLock:
    """Hardened temporal synchronization lock for a single peer link.

    Parameters
    ----------
    node_id : str
        Local node identifier.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TPS estimation
        self.tps_ema: float = 10.0
        self.tps_history: deque[float] = deque(maxlen=TPS_HISTORY_LEN)
        self.alpha: float = 0.1

        # Temporal state
        self.seq_pos: int = 0
        self.τ_history: deque[float] = deque(maxlen=DESYNC_WINDOW)
        self.prev_τ_hash: Optional[str] = None

        # Cumulative desync
        self.Δτ_buffer: deque[float] = deque(maxlen=DESYNC_WINDOW)
        self.integral_Δτ: float = 0.0

        # Probation
        self.probation_until: int = 0
        self.consecutive_waits: int = 0

        # Reed-Solomon codec (2 parity symbols)
        self.rs = RSCodec(RS_NSYM)

        # Timestamp sync (drift correction)
        self.last_timestamp_sync: int = 0
        self.system_clock_offset: float = 0.0

        # Cached generator for unitary wait spin
        self._J: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # TPS estimation with clipping and volatility fallback
    # ------------------------------------------------------------------

    def update_tps(self, measured_tps: float) -> float:
        """Update TPS estimate with clipped EMA and volatility fallback.

        Returns the current tps_ema after update.
        """
        self.tps_history.append(measured_tps)

        # Clipped EMA
        raw_ema = self.alpha * measured_tps + (1 - self.alpha) * self.tps_ema
        self.tps_ema = max(TPS_CLIP_MIN, min(TPS_CLIP_MAX, raw_ema))

        # Volatility check — use median when TPS is noisy
        if len(self.tps_history) >= TPS_HISTORY_LEN:
            std = float(np.std(list(self.tps_history)))
            if std > TPS_VOLATILITY_THRESHOLD:
                self.tps_ema = float(np.median(list(self.tps_history)))

        return self.tps_ema

    # ------------------------------------------------------------------
    # Desync monitoring (cumulative integral)
    # ------------------------------------------------------------------

    def update_desync(self, Δτ: float, num_nodes: int = 2) -> bool:
        """Track signed Δτ and return True if cumulative integral exceeds
        adaptive sever threshold."""
        self.Δτ_buffer.append(Δτ)
        self.integral_Δτ = sum(self.Δτ_buffer)

        threshold = self._adaptive_threshold(num_nodes)
        return abs(self.integral_Δτ) > threshold

    def _adaptive_threshold(self, num_nodes: int = 2) -> float:
        base = DESYNC_BASE_THRESHOLD
        extra = 0.05 * np.log2(num_nodes) if num_nodes > 2 else 0.0
        return min(DESYNC_MAX_THRESHOLD, base + extra)

    # ------------------------------------------------------------------
    # Sequence continuity with hashing
    # ------------------------------------------------------------------

    def record_τ(self, τ_value: float) -> None:
        """Record a new τ_sync value and update the chain hash."""
        self.τ_history.append(τ_value)
        self.seq_pos += 1
        self.prev_τ_hash = self.compute_τ_hash()

    def compute_τ_hash(self) -> Optional[str]:
        """SHA-256 hash of last two τ values for chain validation."""
        if len(self.τ_history) < 2:
            return None
        data = np.array(list(self.τ_history)[-2:], dtype=np.float64).tobytes()
        return hashlib.sha256(data).hexdigest()

    def validate_τ_chain(
        self, received_prev_hash: Optional[str],
    ) -> bool:
        """Return True if the received hash matches our computed chain hash."""
        if received_prev_hash is None:
            return True  # initial step
        computed = self.compute_τ_hash()
        return computed == received_prev_hash

    def handle_jump(
        self, expected_seq: int, received_seq: int,
    ) -> Tuple[bool, bool]:
        """Evaluate a sequence gap.

        Returns
        -------
        (accept, request_missing)
            accept : True if the token can be used immediately.
            request_missing : True if a recovery request should be sent.
        """
        jump = received_seq - expected_seq
        if jump <= 1:
            return True, False  # normal
        if jump <= SEQUENCE_MAX_RECOVERABLE_JUMP:
            return False, True  # request missing shards
        return False, False  # severe jump — reject

    # ------------------------------------------------------------------
    # Unitary wait spin with Cayley fallback and entropy check
    # ------------------------------------------------------------------

    def unitary_wait_spin(
        self,
        h: torch.Tensor,
        t_wait: float,
        mode: str = "auto",
    ) -> torch.Tensor:
        """Apply a unitary evolution to hidden state *h* during wait.

        Parameters
        ----------
        h : Tensor [batch, seq, dim] or [batch, dim]
            Hidden state to evolve.
        t_wait : float
            Wait duration in seconds (governs rotation angle).
        mode : str
            ``'auto'``, ``'cayley'``, or ``'pade'``.

        Returns evolved h. Raises RuntimeError if entropy drift exceeds
        the safety limit.
        """
        if t_wait <= 0:
            return h

        if mode == "auto":
            use_cayley = (t_wait > 0.1) or (not torch.cuda.is_available())
        else:
            use_cayley = mode == "cayley"

        dim = h.shape[-1]

        with torch.no_grad():
            # Build skew-Hermitian generator (cached)
            if self._J is None or self._J.shape[-1] != dim:
                H = torch.randn(dim, dim, device=h.device, dtype=h.dtype)
                self._J = 0.5 * (H - H.T)  # skew-symmetric → exp is orthogonal

            if use_cayley:
                # Cayley 2nd-order: U ≈ I + Jt - (Jt)²/2
                Jt = self._J * t_wait
                U = (
                    torch.eye(dim, device=h.device, dtype=h.dtype)
                    + Jt
                    - (Jt @ Jt) / 2.0
                )
            else:
                # Padé via matrix_exp for small dims, Taylor otherwise
                if dim <= 128:
                    U = torch.matrix_exp(self._J * t_wait)
                else:
                    Jt = self._J * t_wait
                    I = torch.eye(dim, device=h.device, dtype=h.dtype)
                    Jt2 = Jt @ Jt
                    Jt3 = Jt2 @ Jt
                    Jt4 = Jt3 @ Jt
                    U = I + Jt + Jt2 / 2.0 + Jt3 / 6.0 + Jt4 / 24.0

            # Apply to h
            orig_shape = h.shape
            h_flat = h.reshape(-1, dim)

            S_before = self._entropy(h_flat)

            h_evolved = (U @ h_flat.T).T
            h_evolved = h_evolved.reshape(orig_shape)

            S_after = self._entropy(h_evolved.reshape(-1, dim))
            if abs(S_after - S_before) > ENTROPY_DRIFT_LIMIT:
                raise RuntimeError(
                    f"Entropy drift {abs(S_after - S_before):.2e} exceeded "
                    f"limit {ENTROPY_DRIFT_LIMIT:.0e} during unitary wait"
                )

        return h_evolved

    @staticmethod
    def _entropy(h_flat: torch.Tensor) -> float:
        """Spectral entropy proxy via SVD."""
        s = torch.linalg.svdvals(h_flat.float())
        p = s / (s.sum() + 1e-12)
        return -(p * torch.log(p + 1e-10)).sum().item()

    # ------------------------------------------------------------------
    # Probation management
    # ------------------------------------------------------------------

    def check_probation(self, Δτ: float) -> bool:
        """Update consecutive wait counter. Returns True if node should
        be demoted to observer mode."""
        if abs(Δτ) > PROBATION_WAIT_THRESHOLD:
            self.consecutive_waits += 1
        else:
            self.consecutive_waits = max(0, self.consecutive_waits - 1)

        if self.consecutive_waits >= PROBATION_CONSECUTIVE:
            self.probation_until = self.seq_pos + PROBATION_TOKEN_PENALTY
            self.consecutive_waits = 0
            return True
        return False

    def is_on_probation(self) -> bool:
        return self.seq_pos < self.probation_until

    # ------------------------------------------------------------------
    # Temporal shard encoding / decoding (Reed-Solomon)
    # ------------------------------------------------------------------

    def encode_shard(self) -> bytes:
        """Encode temporal state into an RS-protected shard.

        Layout (256 bytes data):
          [0:8]   seq_pos   (uint64 big-endian)
          [8:12]  τ_sync    (float32)
          [12:16] tps_ema   (float32)
          [16:80] prev_τ_hash (64 bytes UTF-8 hex, zero-padded)
          [80:84] integral_Δτ (float32)
          [84:256] zero-pad

        Encoded with RS(nsym=2) per 64-byte symbol → 4 blocks of 66 bytes
        = 264 bytes total.
        """
        data = bytearray(SHARD_DATA_LEN)

        # seq_pos
        data[0:8] = self.seq_pos.to_bytes(8, "big")
        # τ_sync (last recorded)
        τ_val = self.τ_history[-1] if self.τ_history else 0.0
        data[8:12] = np.float32(τ_val).tobytes()
        # tps_ema
        data[12:16] = np.float32(self.tps_ema).tobytes()
        # prev_τ_hash (64 hex chars)
        if self.prev_τ_hash:
            h_bytes = self.prev_τ_hash.encode("ascii")[:64]
            data[16 : 16 + len(h_bytes)] = h_bytes
        # integral_Δτ
        data[80:84] = np.float32(self.integral_Δτ).tobytes()

        # Split into 4 × 64-byte symbols and RS-encode each
        encoded_parts: list[bytes] = []
        for i in range(0, SHARD_DATA_LEN, SHARD_SYMBOL_LEN):
            symbol = bytes(data[i : i + SHARD_SYMBOL_LEN])
            encoded_parts.append(bytes(self.rs.encode(symbol)))

        return b"".join(encoded_parts)

    def decode_shard(
        self, raw_shard: bytes,
    ) -> Tuple[int, float, float, Optional[str], float]:
        """Decode an RS-protected temporal shard.

        Returns (seq_pos, τ_sync, tps_ema, prev_τ_hash, integral_Δτ).
        Raises ValueError on RS decode failure or malformed shard.
        """
        expected_block = SHARD_SYMBOL_LEN + RS_NSYM  # 66 bytes
        num_blocks = SHARD_DATA_LEN // SHARD_SYMBOL_LEN  # 4

        if len(raw_shard) != expected_block * num_blocks:
            raise ValueError(
                f"Invalid shard size: {len(raw_shard)}, "
                f"expected {expected_block * num_blocks}"
            )

        decoded = bytearray()
        for i in range(num_blocks):
            block = raw_shard[i * expected_block : (i + 1) * expected_block]
            try:
                result = self.rs.decode(block)
            except ReedSolomonError as exc:
                raise ValueError(f"RS decoding failed: {exc}") from exc
            # reedsolo.decode returns (decoded_msg, decoded_msgecc, errata_pos)
            decoded.extend(result[0])

        seq = int.from_bytes(decoded[0:8], "big")
        τ = float(np.frombuffer(bytes(decoded[8:12]), dtype=np.float32)[0])
        tps = float(np.frombuffer(bytes(decoded[12:16]), dtype=np.float32)[0])
        raw_hash = bytes(decoded[16:80])
        prev_hash: Optional[str] = raw_hash.decode("ascii").rstrip("\x00") or None
        integral = float(
            np.frombuffer(bytes(decoded[80:84]), dtype=np.float32)[0]
        )
        return seq, τ, tps, prev_hash, integral

    # ------------------------------------------------------------------
    # Periodic timestamp gossip (drift correction)
    # ------------------------------------------------------------------

    def should_timestamp_sync(self) -> bool:
        """Return True if it's time for a timestamp gossip round."""
        return (self.seq_pos - self.last_timestamp_sync) >= TIMESTAMP_GOSSIP_INTERVAL

    def prepare_timestamp_msg(self) -> dict:
        """Build a timestamp gossip message."""
        return {
            "type": "TIMESTAMP",
            "ts": time.monotonic(),
            "node": self.node_id,
        }

    def apply_timestamp_responses(
        self, responses: dict[str, dict],
    ) -> float:
        """Compute clock offset from peer timestamp responses.

        Parameters
        ----------
        responses : dict mapping node_id → {"ts": float}

        Returns the estimated offset in seconds.
        """
        now = time.monotonic()
        offsets: list[float] = []
        for nid, data in responses.items():
            if nid != self.node_id:
                offset = (data["ts"] - now) / 2.0
                offsets.append(offset)
        if offsets:
            self.system_clock_offset = float(np.mean(offsets))
        self.last_timestamp_sync = self.seq_pos
        return self.system_clock_offset
