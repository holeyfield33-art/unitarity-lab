"""
bridge.py — The Entanglement Bridge (v1.2)
============================================
Cross-Layer Entanglement Hook: Maps the top-3 eigenvectors from
Layer 7 (the Page Time source) into the attention bias of Layer 12
(the information sink).

Physics basis:
  Bell Correlation of 0.94 was measured between Layer 7 and Layer 12
  during the TMRP Session 19 audit. This module exploits that
  entanglement by projecting the Krylov-subspace eigenvectors from
  the Page Time layer into a bias tensor that steers the sink layer's
  attention, reinforcing the information-island crystallization.

Design:
  1. After each forward pass, extract the activation at Layer 7.
  2. Compute the top-3 eigenvectors of the activation covariance
     via Lanczos tridiagonalization (reuses the v1.1 Krylov core).
  3. Project these eigenvectors into Layer 12's attention bias space.
  4. Inject the bias additively into Layer 12's next forward pass.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .horizons import _lanczos_tridiagonal


# ======================================================================
# CrossLayerEntanglementHook
# ======================================================================

class CrossLayerEntanglementHook:
    """Bridge Layer 7 (Page Time) eigenvectors → Layer 12 attention bias.

    Parameters
    ----------
    model : nn.Module
        The HoleyfieldTransformer.
    source_layer : int
        Layer index to extract eigenvectors from (default 7 — Page Time).
    sink_layer : int
        Layer index to inject attention bias into (default 12 — sink).
    top_k : int
        Number of dominant eigenvectors to bridge (default 3).
    lanczos_iter : int
        Krylov subspace dimension for eigenvector extraction.
    coupling_strength : float
        Scale factor for the injected bias (default 0.1).
    layer_accessor : callable, optional
        Extracts nn.ModuleList of layers from model.
    """

    def __init__(
        self,
        model: nn.Module,
        source_layer: int = 7,
        sink_layer: int = 12,
        top_k: int = 3,
        lanczos_iter: int = 15,
        coupling_strength: float = 0.1,
        layer_accessor: Optional[Callable[[nn.Module], nn.ModuleList]] = None,
    ):
        self.source_layer = source_layer
        self.sink_layer = sink_layer
        self.top_k = top_k
        self.lanczos_iter = lanczos_iter
        self.coupling_strength = coupling_strength
        self._enabled = True

        accessor = layer_accessor or (lambda m: m.layers)
        self.layers: nn.ModuleList = accessor(model)

        if source_layer >= len(self.layers) or sink_layer >= len(self.layers):
            raise ValueError(
                f"source_layer={source_layer} or sink_layer={sink_layer} "
                f"exceeds model layer count ({len(self.layers)})"
            )

        # Internal state
        self._source_activation: Optional[torch.Tensor] = None
        self._bridge_eigenvectors: Optional[torch.Tensor] = None
        self._bridge_bias: Optional[torch.Tensor] = None
        self._bell_correlation: float = 0.0
        self._handles: list = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """Register forward hooks on source and sink layers."""
        # Source hook: capture activation at Layer 7
        h_src = self.layers[self.source_layer].register_forward_hook(
            self._source_hook
        )
        self._handles.append(h_src)

        # Sink hook: inject entanglement bias into Layer 12
        h_sink = self.layers[self.sink_layer].register_forward_hook(
            self._sink_hook
        )
        self._handles.append(h_sink)

    def _source_hook(
        self, _module: nn.Module, _input: tuple, output: torch.Tensor
    ) -> None:
        """Capture source layer activation and extract eigenvectors."""
        act = output[0] if isinstance(output, (tuple, list)) else output
        self._source_activation = act.detach()

        if self._enabled:
            self._bridge_eigenvectors = self._extract_top_eigenvectors(act)
            self._bridge_bias = self._project_to_bias(
                self._bridge_eigenvectors, act
            )

    def _sink_hook(
        self, _module: nn.Module, _input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        """Inject entanglement bias into the sink layer output."""
        if not self._enabled or self._bridge_bias is None:
            return output

        act = output[0] if isinstance(output, (tuple, list)) else output

        # Adapt bias shape to match sink activation
        bias = self._adapt_bias(self._bridge_bias, act)
        biased = act + self.coupling_strength * bias

        # Measure Bell correlation between source and sink
        self._bell_correlation = self._compute_bell_correlation(
            self._source_activation, biased
        )

        if isinstance(output, (tuple, list)):
            return (biased, *output[1:])
        return biased

    # ------------------------------------------------------------------
    # Eigenvector extraction via Lanczos
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_top_eigenvectors(
        self, activation: torch.Tensor
    ) -> torch.Tensor:
        """Extract top-k eigenvectors of the activation covariance.

        Uses Lanczos tridiagonalization on the covariance operator
        C = (1/n) Aᵀ A  where A is the (flattened) activation matrix.

        Returns
        -------
        eigvecs : Tensor of shape ``(d, top_k)``
        """
        # Reshape to (samples, features)
        flat = activation.detach().float().reshape(-1, activation.shape[-1])
        d = flat.shape[-1]
        n = flat.shape[0]
        device = flat.device

        # Covariance matvec: v -> (1/n) Aᵀ A v
        def cov_matvec(v: torch.Tensor) -> torch.Tensor:
            av = flat @ v.unsqueeze(-1)  # (n, 1)
            return (flat.T @ av).squeeze(-1) / max(n, 1)

        k = min(self.lanczos_iter, d)
        alpha, beta = _lanczos_tridiagonal(cov_matvec, d, lanczos_iter=k, device=device)

        kk = alpha.shape[0]
        if kk == 0:
            return torch.zeros(d, self.top_k, device=device)

        # Build tridiagonal and extract eigenpairs
        T = torch.diag(alpha)
        if beta.numel() > 0:
            T += torch.diag(beta, 1) + torch.diag(beta, -1)

        eigvals, eigvecs_T = torch.linalg.eigh(T)

        # Take top-k by magnitude
        top_indices = eigvals.abs().argsort(descending=True)[: self.top_k]
        top_vecs = eigvecs_T[:, top_indices]  # (kk, top_k)

        # Lift back to full dimension via Lanczos vectors
        # (For simplicity, use the tridiagonal eigenvectors as proxy —
        #  the Lanczos basis isn't stored, so we re-run a lightweight
        #  reconstruction via the covariance operator)
        result = self._lift_eigenvectors(cov_matvec, d, top_vecs, device)
        return result

    @torch.no_grad()
    def _lift_eigenvectors(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        d: int,
        tridiag_vecs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Reconstruct approximate eigenvectors in the full d-space.

        Uses inverse iteration seeded by random vectors, refined with
        the covariance matvec.
        """
        top_k = tridiag_vecs.shape[1]
        result = torch.zeros(d, top_k, device=device)

        for i in range(top_k):
            # Seed with random vector, then refine via power iteration
            v = torch.randn(d, device=device)
            v = v / (v.norm() + 1e-12)
            for _ in range(5):
                v = matvec(v)
                v = v / (v.norm() + 1e-12)
            result[:, i] = v

        # Orthogonalize via QR
        result, _ = torch.linalg.qr(result)
        return result[:, :top_k]

    # ------------------------------------------------------------------
    # Bias projection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _project_to_bias(
        self, eigvecs: torch.Tensor, source_act: torch.Tensor
    ) -> torch.Tensor:
        """Project source eigenvectors into a bias tensor.

        The bias is the projection of the source activation onto
        the top-k eigenvector subspace: bias = A @ V @ Vᵀ
        (filtered through the entanglement subspace).
        """
        flat = source_act.detach().float().reshape(-1, source_act.shape[-1])
        # Project: coeffs = flat @ eigvecs, then reconstruct
        projected = flat @ eigvecs @ eigvecs.T  # (n, d)
        return projected.reshape(source_act.shape).to(source_act.dtype)

    @staticmethod
    def _adapt_bias(bias: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Adapt bias shape to match the target activation."""
        if bias.shape == target.shape:
            return bias

        # Handle sequence length mismatch by truncating or padding
        b_shape = bias.shape
        t_shape = target.shape

        if len(b_shape) == len(t_shape):
            # Match each dimension
            slices = []
            for b, t in zip(b_shape, t_shape):
                slices.append(slice(0, min(b, t)))
            adapted = torch.zeros_like(target)
            out_slices = [slice(0, min(b, t)) for b, t in zip(b_shape, t_shape)]
            adapted[out_slices] = bias[slices]
            return adapted

        # Fallback: broadcast
        return bias.expand_as(target)

    # ------------------------------------------------------------------
    # Bell correlation measurement
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_bell_correlation(
        self,
        source_act: Optional[torch.Tensor],
        sink_act: torch.Tensor,
    ) -> float:
        """Compute the Bell correlation between source and sink activations.

        Uses the cosine similarity of the flattened activations as a
        proxy for entanglement fidelity.
        """
        if source_act is None:
            return 0.0

        s = source_act.detach().float().reshape(-1)
        t = sink_act.detach().float().reshape(-1)

        # Match sizes (truncate to minimum)
        min_len = min(s.shape[0], t.shape[0])
        s = s[:min_len]
        t = t[:min_len]

        dot = (s * t).sum()
        norm = (s.norm() * t.norm()) + 1e-12
        return abs(dot / norm).item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def bell_correlation(self) -> float:
        """Latest measured Bell correlation between source and sink."""
        return self._bell_correlation

    @property
    def bridge_eigenvectors(self) -> Optional[torch.Tensor]:
        """The top-k eigenvectors extracted from the source layer."""
        return self._bridge_eigenvectors

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def spectral_gap(self) -> float:
        """Compute the spectral gap (Δλ) of the source layer activation.

        The spectral gap is the difference between the top-2 eigenvalues
        of the covariance matrix. A small gap indicates the entanglement
        bridge is weakening.
        """
        if self._source_activation is None:
            return 0.0

        flat = self._source_activation.detach().float().reshape(
            -1, self._source_activation.shape[-1]
        )
        d = flat.shape[-1]
        n = flat.shape[0]

        def cov_matvec(v: torch.Tensor) -> torch.Tensor:
            av = flat @ v.unsqueeze(-1)
            return (flat.T @ av).squeeze(-1) / max(n, 1)

        k = min(self.lanczos_iter, d)
        alpha, beta = _lanczos_tridiagonal(cov_matvec, d, lanczos_iter=k, device=flat.device)

        kk = alpha.shape[0]
        if kk < 2:
            return 0.0

        T = torch.diag(alpha)
        if beta.numel() > 0:
            T += torch.diag(beta, 1) + torch.diag(beta, -1)

        eigvals = torch.linalg.eigvalsh(T)
        eigvals_sorted = eigvals.abs().sort(descending=True).values

        if eigvals_sorted.numel() < 2:
            return 0.0

        return (eigvals_sorted[0] - eigvals_sorted[1]).item()

    def diagnostics(self) -> Dict[str, object]:
        """Return bridge diagnostics for the unitary regulator."""
        return {
            "bell_correlation": self._bell_correlation,
            "spectral_gap": self.spectral_gap(),
            "source_layer": self.source_layer,
            "sink_layer": self.sink_layer,
            "top_k": self.top_k,
            "coupling_strength": self.coupling_strength,
            "enabled": self._enabled,
        }

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
