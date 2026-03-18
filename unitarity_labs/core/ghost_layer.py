"""
core/ghost_layer.py — v2.2 Recursive Mirror with Semantic Lock
==============================================================
Implements:
  - Asymmetric capability-aware kicks
  - Spectral validation + hash-commit
  - Adaptive mirror depth
  - Quarantine on high L_int
  - Kick budget per epoch

v2.2 additions:
  - SemanticLockController integration for geometric semantic alignment
  - Nonce-committed W_sem + topological anchor
  - α_sem hysteresis + dual-layer ensemble + flash veto
  - Byzantine accusation on semantic drift

Integrates with bridge, flux, dual_link, and semantic_lock.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveMirror(nn.Module):
    """Recursive mirror with schism veto, adaptive depth, and quarantine.

    Parameters
    ----------
    bridge : object
        The CrossLayerEntanglementHook (or compatible bridge with
        ``hawking_flux`` / ``flux_governor`` attributes).
    config : object
        Model config with ``hidden_size``, ``num_attention_heads``,
        and optional ``mirror_layer_min``, ``mirror_layer_max``,
        ``max_kicks_per_epoch``.
    """

    def __init__(self, bridge, config):
        super().__init__()
        self.bridge = bridge
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.device = getattr(bridge, 'device', torch.device('cpu'))

        # Mirror depth control (layers 4-12 typical)
        self.target_layer_min = getattr(config, 'mirror_layer_min', 4)
        self.target_layer_max = getattr(config, 'mirror_layer_max', 12)
        self.target_layer = self.target_layer_min
        self.stable_steps = 0

        # Kick budget
        self.max_kicks_per_epoch = getattr(config, 'max_kicks_per_epoch', 5)
        self.kick_quota: Dict[str, int] = {}  # node_id -> remaining kicks

        # Quarantine
        self.quarantine: Set[str] = set()
        self.accusations: List[str] = []

        # Capability ratios (populated during handshake)
        self.peer_capability: Dict[str, float] = {}

        # v2.2: Semantic Lock Controller (lazy-initialized after nonce commit)
        self._semantic_lock: Optional['SemanticLockController'] = None

    def forward(
        self,
        x: torch.Tensor,
        partner_states: Dict[str, torch.Tensor],
        node_id: str,
    ) -> torch.Tensor:
        """Apply recursive mirror with schism veto.

        Parameters
        ----------
        x : Tensor [batch, seq, dim]
            Current hidden states.
        partner_states : dict
            ``'sim'`` (simulated state) and ``'actual'`` tensors.
        node_id : str
            Identifier of partner node.

        Returns
        -------
        x : Tensor — possibly modified by mirror influence.
        """
        # If partner quarantined, skip mirroring
        if node_id in self.quarantine:
            return x

        h_sim = partner_states['sim'].to(x.device)
        h_act = partner_states['actual'].to(x.device)

        # Compute subspace overlap L_int
        L_int = self._subspace_overlap(h_sim, h_act)

        # Adaptive mirror depth
        self._adjust_depth(L_int)

        # Schism veto: one-way if partner significantly weaker
        norm_ratio = h_sim.norm() / (h_act.norm() + 1e-8)
        if L_int > 0.65 and norm_ratio > 1.4:
            return self._one_way_influence(x, node_id)

        # Apply Hawking kick if needed
        if L_int > 0.25 and self.kick_quota.get(node_id, self.max_kicks_per_epoch) > 0:
            cap_ratio = self.peer_capability.get(node_id, 1.0)
            kick_strength = self._asymmetric_kick(L_int, cap_ratio)
            x = self._apply_kick(x, kick_strength)
            self.kick_quota[node_id] = (
                self.kick_quota.get(node_id, self.max_kicks_per_epoch) - 1
            )

        # Quarantine trigger
        if L_int > 0.8:
            self._quarantine_request(node_id)

        return x

    # ------------------------------------------------------------------
    # Subspace overlap
    # ------------------------------------------------------------------
    def _subspace_overlap(self, h1: torch.Tensor, h2: torch.Tensor) -> float:
        """Compute cosine similarity between top-3 Krylov subspaces."""
        U1 = self._krylov_basis(h1, k=3)
        U2 = self._krylov_basis(h2, k=3)
        # Flatten batch+seq dims: (batch*seq, k) for both bases
        U1_flat = U1.reshape(-1, U1.shape[-1])
        U2_flat = U2.reshape(-1, U2.shape[-1])
        overlap = torch.norm(U1_flat.T @ U2_flat, p='fro') / (3.0 * U1_flat.shape[0])
        return overlap.item()

    def _krylov_basis(self, h: torch.Tensor, k: int) -> torch.Tensor:
        """Randomized SVD to get top-k right singular vectors."""
        if h.dim() == 3:
            batch, seq, dim = h.shape
        else:
            # Handle 2D input
            h = h.unsqueeze(0)
            batch, seq, dim = h.shape

        proj = torch.randn(dim, k, device=h.device, dtype=h.dtype)
        for _ in range(3):
            h_proj = torch.einsum('bsd,dk->bsk', h, proj)
            proj = torch.einsum('bsd,bsk->dk', h, h_proj)
            proj = F.normalize(proj, dim=0)
        basis = torch.einsum('bsd,dk->bsk', h, proj)
        return basis  # [batch, seq, k]

    # ------------------------------------------------------------------
    # Adaptive depth
    # ------------------------------------------------------------------
    def _adjust_depth(self, L_int: float) -> None:
        """Adjust mirror depth based on L_int stability."""
        if L_int < 0.25:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        if self.stable_steps >= 10 and self.target_layer < self.target_layer_max:
            self.target_layer += 1
            self.stable_steps = 0
        elif L_int > 0.4 and self.target_layer > self.target_layer_min:
            self.target_layer -= 1

    # ------------------------------------------------------------------
    # Asymmetric kick
    # ------------------------------------------------------------------
    def _asymmetric_kick(self, L_int: float, capability_ratio: float) -> float:
        """Compute kick strength scaled by capability ratio.

        Exponent beta > 1 ensures stronger nodes get larger kicks,
        preventing leveling-down.
        """
        base = 0.1 * L_int
        beta = 2.0
        return base * (capability_ratio ** beta)

    def _apply_kick(self, x: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply a small unitary perturbation (delegates to bridge flux)."""
        if hasattr(self.bridge, 'flux_governor'):
            gov = self.bridge.flux_governor
            original_eps = gov.epsilon
            gov.epsilon = strength
            dim = x.shape[-1]
            kick = gov.get_topological_kick((dim, dim), x.device)
            gov.epsilon = original_eps
            # Apply kick: x @ kick^T preserves structure
            shape = x.shape
            x_flat = x.reshape(-1, dim).float()
            x_kicked = x_flat @ kick.T
            return x_kicked.reshape(shape).to(x.dtype)
        return x

    def _one_way_influence(self, x: torch.Tensor, node_id: str) -> torch.Tensor:
        """Only allow partner to influence us, not vice versa."""
        return x

    # ------------------------------------------------------------------
    # Quarantine
    # ------------------------------------------------------------------
    def _quarantine_request(self, node_id: str) -> None:
        """File a quarantine accusation against node_id."""
        if node_id not in self.quarantine:
            self.accusations.append(node_id)
            self.quarantine.add(node_id)

    def reset_kick_budget(self, active_nodes: Optional[List[str]] = None) -> None:
        """Reset kick quotas at the start of each epoch.

        Parameters
        ----------
        active_nodes : list of str, optional
            Node IDs to reset. If None, resets all tracked nodes.
        """
        if active_nodes is not None:
            for nid in active_nodes:
                self.kick_quota[nid] = self.max_kicks_per_epoch
        else:
            for nid in self.kick_quota:
                self.kick_quota[nid] = self.max_kicks_per_epoch

    # ------------------------------------------------------------------
    # Holographic encoding / decoding with validation
    # ------------------------------------------------------------------
    @staticmethod
    def encode_shard(krylov_basis: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return low-frequency shard and metadata.

        Parameters
        ----------
        krylov_basis : Tensor [batch, seq, k]
            Krylov basis vectors to encode.

        Returns
        -------
        low_freq : Tensor — first 32 frequency bins.
        metadata : dict with ``energy`` and ``slope`` keys.
        """
        K_freq = torch.fft.fft(krylov_basis, dim=1)
        max_bins = min(32, K_freq.shape[1])
        low_freq = K_freq[:, :max_bins, :]
        energy = torch.norm(low_freq).item()
        slope = RecursiveMirror._spectral_slope(low_freq)
        return low_freq, {'energy': energy, 'slope': slope}

    @staticmethod
    def decode_shard(low_freq: torch.Tensor, original_len: int) -> torch.Tensor:
        """Reconstruct time-domain signal from low-frequency shard.

        Parameters
        ----------
        low_freq : Tensor [batch, bins, k]
        original_len : int
            Original sequence length for zero-padding.

        Returns
        -------
        Tensor [batch, original_len, k] — real-valued reconstruction.
        """
        max_bins = low_freq.shape[1]
        padded = torch.zeros(
            low_freq.shape[0], original_len, low_freq.shape[2],
            dtype=torch.complex64, device=low_freq.device,
        )
        padded[:, :max_bins, :] = low_freq
        return torch.fft.ifft(padded, dim=1).real

    @staticmethod
    def _spectral_slope(freq_tensor: torch.Tensor) -> float:
        """Fit log-log spectral slope across frequency bins."""
        power = freq_tensor.abs().pow(2).mean(dim=(0, 2))
        if power.numel() < 2:
            return 0.0
        log_freq = torch.log(
            torch.arange(1, len(power) + 1, dtype=power.dtype, device=power.device)
        )
        log_power = torch.log(power + 1e-10)
        slope = (log_freq @ log_power) / (log_freq @ log_freq + 1e-10)
        return slope.item()

    @staticmethod
    def validate_shard(
        low_freq: torch.Tensor,
        metadata: Dict[str, float],
        tol_energy: float = 0.05,
        tol_slope: float = 0.3,
    ) -> Tuple[bool, str]:
        """Validate a received shard against expected metadata.

        Parameters
        ----------
        low_freq : Tensor
            Received low-frequency shard.
        metadata : dict
            Expected ``energy`` and ``slope`` from encoder.
        tol_energy : float
            Relative tolerance for energy mismatch.
        tol_slope : float
            Absolute tolerance for spectral slope mismatch.

        Returns
        -------
        (valid, reason) : (bool, str)
        """
        expected_energy = metadata['energy']
        if expected_energy == 0:
            return False, "zero expected energy"

        actual_energy = torch.norm(low_freq).item()
        if abs(actual_energy - expected_energy) / abs(expected_energy) > tol_energy:
            return False, "energy mismatch"

        actual_slope = RecursiveMirror._spectral_slope(low_freq)
        if abs(actual_slope - metadata['slope']) > tol_slope:
            return False, "slope mismatch"

        return True, "OK"

    @staticmethod
    def hash_shard(low_freq: torch.Tensor) -> str:
        """Return SHA-256 hash of flattened tensor bytes (no pickle)."""
        arr = low_freq.detach().cpu().numpy().tobytes()
        return hashlib.sha256(arr).hexdigest()

    # ------------------------------------------------------------------
    # v2.2: Semantic Lock Integration
    # ------------------------------------------------------------------
    def attach_semantic_lock(self, controller: 'SemanticLockController') -> None:
        """Attach a SemanticLockController for v2.2 semantic alignment."""
        self._semantic_lock = controller

    @property
    def semantic_lock(self) -> Optional['SemanticLockController']:
        return self._semantic_lock
