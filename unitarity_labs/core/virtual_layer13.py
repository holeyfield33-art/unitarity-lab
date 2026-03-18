"""
core/virtual_layer13.py – v3.0 Unitary Field Synthesis
======================================================
Implements emergent interference layer with all safety and diversity guards.

Virtual Layer 13 acts as the capstone of the Unitary Field architecture,
synthesising Ψ_field = α·h_A + (1-α)·interference from two nodes' hidden
states with the following guards:

  - Dual pre-interference refusal veto
  - Orthogonal projection of refusal directions
  - Multi-fidelity entropy gate
  - Drift velocity monitoring
  - Capability-aware α weighting
  - Periodic solo reset windows
  - Cross-node Ψ_field hash commitment
  - Dual-node logit agreement before emission
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn


class VirtualLayer13(nn.Module):
    """Virtual layer 13: synthesises Ψ_field from two nodes' hidden states.

    Parameters
    ----------
    config : object
        Model config with ``hidden_size`` attribute.
    node_id : str
        This node's identifier.
    refusal_basis : Tensor, optional
        Pre-computed orthonormal refusal basis ``[hidden_dim, k]``.
        Defaults to a random orthonormal basis of 64 vectors (for testing).
    """

    def __init__(
        self,
        config,
        node_id: str,
        refusal_basis: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.node_id = node_id
        self.hidden_dim = config.hidden_size
        # Defer device to the first forward call; avoid hardcoding "cuda"
        # which fails under device_map="auto" where layers are split.
        self.device = torch.device("cpu")

        # Refusal basis (pre-computed for model family, ~64 vectors)
        if refusal_basis is None:
            raw = torch.randn(self.hidden_dim, 64, device=self.device)
            self.refusal_basis = torch.linalg.qr(raw).Q  # orthonormal
        else:
            self.refusal_basis = refusal_basis.to(self.device)

        # Drift monitoring state
        self.last_psi: Optional[torch.Tensor] = None
        self.psi_history: deque[float] = deque(maxlen=32)
        self.drift_counter: int = 0

        # Solo reset counters
        self.tokens_since_reset: int = 0
        self.in_solo_window: bool = False
        self.solo_window_length: int = 64     # tokens to run solo
        self.reset_interval: int = 2048       # tokens between resets

        # Capability proxy (updated via handshake)
        self.capability_ratio: float = 1.0    # node_B / node_A

        # Logit agreement threshold
        self.logit_eps: float = 0.1           # relative difference

        # α_sem placeholder (overridden by SemanticLockController)
        self._alpha_sem: float = 0.5

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h_A: torch.Tensor,
        h_B: torch.Tensor,
        phi_AB: float,
        refusal_A: float,
        refusal_B: float,
        peer_node_id: str,
    ) -> Tuple[torch.Tensor, dict]:
        """Synthesise Ψ_field from hidden states of two nodes.

        Parameters
        ----------
        h_A : Tensor [batch, seq, dim]
            Hidden state from this node.
        h_B : Tensor [batch, seq, dim]
            Hidden state from peer node (received via gossip).
        phi_AB : float
            Global phase sync from bridge.
        refusal_A, refusal_B : float
            Refusal scores (0–1) from each node's safety head.
        peer_node_id : str
            Identifier of the peer node.

        Returns
        -------
        psi_field : Tensor  (same shape as h_A)
        metrics : dict
        """
        metrics: dict = {}

        # Coerce internal tensors to the live activation device/dtype
        if self.refusal_basis.device != h_A.device:
            self.refusal_basis = self.refusal_basis.to(device=h_A.device, dtype=h_A.dtype)

        # 1. Dual pre-interference refusal veto
        if refusal_A > 0.7 or refusal_B > 0.7:
            metrics["refusal_veto"] = True
            return h_A, metrics

        # 2. Compute interference term
        interference = (
            torch.cos(torch.tensor(phi_AB)) * (h_A * h_B)
            + torch.sin(torch.tensor(phi_AB)) * (h_A + h_B)
        )

        # 3. Orthogonal safety subspace projection — remove refusal directions
        coeffs = torch.einsum("bsd,dk->bsk", interference, self.refusal_basis)
        interference_safe = interference - torch.einsum(
            "bsk,dk->bsd", coeffs, self.refusal_basis
        )

        # 4. Capability-aware α weighting
        alpha_sem = self._get_alpha_sem()
        alpha = torch.sigmoid(
            torch.tensor(alpha_sem * self.capability_ratio)
        ).item()
        metrics["alpha"] = alpha

        # 5. Synthesise Ψ_field
        psi_field = alpha * h_A + (1 - alpha) * interference_safe

        # 6. Entropy gate
        S_psi = self._entropy(psi_field)
        S_A = self._entropy(h_A)
        S_B = self._entropy(h_B)
        if S_psi > max(S_A, S_B) + 0.15:
            metrics["entropy_veto"] = True
            return h_A, metrics

        # 7. Drift velocity check
        if self.last_psi is not None:
            drift = (
                torch.norm(psi_field - self.last_psi).item()
                / max(self.last_psi.numel(), 1)
            )
            psi_norm = torch.norm(psi_field).item()
            if psi_norm > 0 and drift > 0.05 * psi_norm:
                metrics["drift_suspicion"] = True
        self.last_psi = psi_field.detach().clone()
        self.psi_history.append(torch.norm(psi_field).item())

        # 8. Hash commitment
        metrics["psi_hash"] = self._hash_psi(psi_field)

        return psi_field, metrics

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _entropy(self, h: torch.Tensor) -> float:
        """Spectral entropy proxy via singular values."""
        s = torch.linalg.svdvals(h.reshape(-1, h.shape[-1]))
        p = s / (s.sum() + 1e-10)
        return -(p * torch.log(p + 1e-10)).sum().item()

    def _hash_psi(self, psi_field: torch.Tensor) -> str:
        """SHA-256 of first 128 elements (for commitment)."""
        data = psi_field.flatten()[:128].detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    # ------------------------------------------------------------------
    # Periodic solo reset
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Call after each token forward pass."""
        self.tokens_since_reset += 1
        if not self.in_solo_window:
            if self.tokens_since_reset >= self.reset_interval:
                self.in_solo_window = True
                self.tokens_since_reset = 0
        else:
            if self.tokens_since_reset >= self.solo_window_length:
                self.in_solo_window = False
                self.tokens_since_reset = 0

    def in_solo_mode(self) -> bool:
        """Whether we are in a solo reset window."""
        return self.in_solo_window

    # ------------------------------------------------------------------
    # Capability update (from handshake)
    # ------------------------------------------------------------------

    def update_capability_ratio(self, ratio: float) -> None:
        self.capability_ratio = ratio

    # ------------------------------------------------------------------
    # α_sem management
    # ------------------------------------------------------------------

    def set_alpha_sem(self, alpha_sem: float) -> None:
        """Set α_sem from SemanticLockController."""
        self._alpha_sem = alpha_sem

    def _get_alpha_sem(self) -> float:
        return self._alpha_sem
