"""
casimir_opt.py — The Topological Pressure Optimizer (v3.0.0-Singularity)
=========================================================================
Replaces standard Adam/SGD with a Casimir Pressure Optimizer.

v1.1 — DeepSeek Integration:
  - **rSVD** (Randomized SVD) for O(n² log k) Casimir projection.
  - Explicit **Kolmogorov -5/3 Regularization** method.
  - **Hamiltonian Invariant Check**: update rejected if β₀ drifts.

Design principles (TMRP Session 18):
  - Penalize "Laminar" weights (redundant, low-rank information).
  - Reward "Turbulent" weight distributions that follow the **-5/3
    Kolmogorov Energy Cascade Law** in their spectral density.
  - Hard Constraint: preserve the Betti numbers of the weight manifold
    during training to ensure **Topological Stability** (the Hamiltonian
    of Invariants).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


# ======================================================================
# Randomized SVD  (rSVD) — O(n² log k) approximation
# ======================================================================

def rsvd(
    weight: torch.Tensor,
    rank: int = 10,
    n_oversamples: int = 5,
    n_power_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD for fast low-rank approximation.

    Approximates the top-*rank* singular triplets of *weight* using
    a randomized range finder + QR + small SVD.

    Complexity: O(n * m * (rank + n_oversamples)) — much cheaper than
    full SVD for large matrices when rank << min(n, m).

    Parameters
    ----------
    weight : Tensor of shape ``(n, m)``
    rank : int — target rank.
    n_oversamples : int — extra columns for better approximation.
    n_power_iter : int — subspace iteration steps.

    Returns
    -------
    U : (n, rank), S : (rank,), V : (m, rank)
    """
    w = weight.detach().float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)

    n, m = w.shape
    k = min(rank + n_oversamples, n, m)

    # Random Gaussian sketch
    omega = torch.randn(m, k, device=w.device, dtype=w.dtype)
    Y = w @ omega  # (n, k)

    # Power iteration for better range approximation
    for _ in range(n_power_iter):
        Y = w @ (w.T @ Y)

    Q, _ = torch.linalg.qr(Y)  # (n, k) orthonormal basis

    # Project into low-rank subspace
    B = Q.T @ w  # (k, m)

    # Small SVD on B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat  # (n, k)

    # Trim to target rank
    r = min(rank, S.shape[0])
    return U[:, :r], S[:r], Vt[:r, :]


# ======================================================================
# Helpers — spectral analysis of weight tensors
# ======================================================================

def _spectral_density(weight: torch.Tensor) -> torch.Tensor:
    """Compute the 1-D power spectral density of a weight matrix."""
    w = weight.detach().float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)

    spectrum = torch.fft.rfft(w, dim=-1)
    psd = (spectrum.real.pow(2) + spectrum.imag.pow(2)).mean(dim=0)
    return psd + 1e-12


def _kolmogorov_penalty(weight: torch.Tensor) -> torch.Tensor:
    r"""Measure deviation from the Kolmogorov -5/3 energy cascade.

    Fits the log-log PSD against a slope of $-5/3$ and returns the
    mean-squared residual.
    """
    psd = _spectral_density(weight)
    n = psd.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=weight.device)

    k = torch.arange(1, n, dtype=torch.float32, device=weight.device)
    log_k = torch.log(k)
    log_psd = torch.log(psd[1:])

    ideal_slope = -5.0 / 3.0
    intercept = (log_psd - ideal_slope * log_k).mean()
    ideal_log_psd = ideal_slope * log_k + intercept

    residual = (log_psd - ideal_log_psd).pow(2).mean()
    return residual


def _laminar_penalty(weight: torch.Tensor) -> torch.Tensor:
    """Penalize laminar (low-rank / redundant) weight structure.

    Uses **rSVD** (v1.1) instead of full SVD for O(n² log k) performance.
    """
    w = weight.detach().float()
    if w.dim() == 1:
        return torch.tensor(0.0, device=weight.device)
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)

    rank = min(10, *w.shape)
    _, sv, _ = rsvd(w, rank=rank)
    sv = sv / (sv.sum() + 1e-12)
    entropy = -(sv * (sv + 1e-12).log()).sum()
    max_entropy = math.log(rank)
    normalized = entropy / (max_entropy + 1e-12)

    return 1.0 - normalized


# ======================================================================
# Betti-number estimator (topological stability constraint)
# ======================================================================

def estimate_betti_0(weight: torch.Tensor, threshold: float = 0.1) -> int:
    """Estimate the 0th Betti number (connected components) of the weight.

    Uses cosine-similarity adjacency + union-find on the row dimension.
    """
    w = weight.detach().float()
    if w.dim() == 1:
        return 1
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)

    norms = w.norm(dim=1, keepdim=True) + 1e-12
    normed = w / norms
    sim = normed @ normed.T

    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j].item() > threshold:
                union(i, j)

    return len({find(i) for i in range(n)})


# ======================================================================
# CasimirOptimizer
# ======================================================================

class CasimirOptimizer(Optimizer):
    """Topological Pressure Optimizer (Casimir variant).

    v1.1 additions:
      - ``_project_to_casimir`` uses **rSVD** for O(n² log k) projection.
      - ``_apply_turbulence_regularization`` enforces Kolmogorov -5/3.
      - ``_hamiltonian_invariant_check`` rejects updates that change β₀.

    Wraps a momentum-based update with two pressure terms:

    1. **Turbulence reward** — Kolmogorov -5/3 cascade penalty.
    2. **Laminar penalty** — penalizes low-rank / redundant structure.

    Hard constraint: β₀ of each parameter must remain constant.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        turbulence_weight: float = 0.01,
        laminar_weight: float = 0.01,
        betti_threshold: float = 0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            turbulence_weight=turbulence_weight,
            laminar_weight=laminar_weight,
            betti_threshold=betti_threshold,
        )
        super().__init__(params, defaults)

        self._initial_betti: Dict[int, int] = {}
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                self._initial_betti[pid] = estimate_betti_0(
                    p.data, group["betti_threshold"]
                )

    # ------------------------------------------------------------------
    # rSVD-based Casimir projection (v1.1)
    # ------------------------------------------------------------------
    def _project_to_casimir(
        self, weight: torch.Tensor, rank: int = 10
    ) -> torch.Tensor:
        """Project a weight matrix into the Casimir subspace via rSVD.

        Removes the top-rank laminar components (dominant singular
        directions), keeping the turbulent residual that better
        follows the Kolmogorov spectrum.

        Returns the projected weight.
        """
        if weight.dim() < 2:
            return weight

        w = weight.detach().float()
        if w.dim() > 2:
            orig_shape = w.shape
            w = w.reshape(w.shape[0], -1)
        else:
            orig_shape = None

        r = min(rank, *w.shape)
        U, S, Vt = rsvd(w, rank=r)

        # Reconstruct low-rank (laminar) component
        laminar = U @ torch.diag(S) @ Vt

        # Casimir-projected weight = turbulent residual
        projected = w - laminar

        if orig_shape is not None:
            projected = projected.reshape(orig_shape)

        return projected.to(weight.dtype)

    # ------------------------------------------------------------------
    # Kolmogorov -5/3 Regularization (v1.1)
    # ------------------------------------------------------------------
    def _apply_turbulence_regularization(
        self, param: torch.nn.Parameter, group: dict
    ) -> torch.Tensor:
        """Return a regularization gradient that pushes the weight spectrum
        toward the Kolmogorov -5/3 law.

        Uses finite-difference estimation of the gradient of the
        Kolmogorov penalty + the laminar penalty.
        """
        tw = group["turbulence_weight"]
        lw = group["laminar_weight"]

        grad = torch.zeros_like(param.data)
        if param.data.dim() < 2:
            return grad

        eps = 1e-5
        noise = torch.randn_like(param.data) * eps

        # Kolmogorov direction
        base_kp = _kolmogorov_penalty(param.data)
        perturbed_kp = _kolmogorov_penalty(param.data + noise)
        grad += tw * ((perturbed_kp - base_kp) / eps) * noise / eps

        # Laminar direction
        base_lp = _laminar_penalty(param.data)
        perturbed_lp = _laminar_penalty(param.data + noise)
        grad += lw * ((perturbed_lp - base_lp) / eps) * noise / eps

        return grad

    # ------------------------------------------------------------------
    # Hamiltonian Invariant Check (v1.1)
    # ------------------------------------------------------------------
    def _hamiltonian_invariant_check(
        self, param_id: int, proposed: torch.Tensor, betti_threshold: float
    ) -> bool:
        """Check that the proposed weight preserves β₀.

        Returns True if the update is safe (β₀ is preserved),
        False if it must be rejected.
        """
        new_betti = estimate_betti_0(proposed, betti_threshold)
        original_betti = self._initial_betti.get(param_id, new_betti)
        return new_betti == original_betti

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            bt = group["betti_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                pid = id(p)
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]

                # Combined gradient: task grad + turbulence regularization
                d_p = p.grad.data + self._apply_turbulence_regularization(p, group)

                buf.mul_(mu).add_(d_p)

                # Tentative update
                proposed = p.data - lr * buf

                # ----- Hamiltonian Invariant Check (β₀ hard constraint) -----
                if not self._hamiltonian_invariant_check(pid, proposed, bt):
                    continue

                p.data.copy_(proposed)

        return loss

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict[str, object]:
        """Return a snapshot of Casimir pressures and Betti numbers."""
        info: Dict[str, object] = {}
        for gidx, group in enumerate(self.param_groups):
            for pidx, p in enumerate(group["params"]):
                key = f"group{gidx}_param{pidx}"
                info[key] = {
                    "kolmogorov_penalty": _kolmogorov_penalty(p.data).item(),
                    "laminar_penalty": _laminar_penalty(p.data).item(),
                    "betti_0_initial": self._initial_betti.get(id(p), -1),
                    "betti_0_current": estimate_betti_0(
                        p.data, group["betti_threshold"]
                    ),
                }
        return info
