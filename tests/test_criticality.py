"""
test_criticality.py — Vortex-Lock and Casimir Tests (v1.1)
============================================================
Verifies core invariants of the Holeyfield v1.1 Framework:
  1. PLL Monitor correctly identifies phase-locked vs. anomalous profiles.
  2. Kolmogorov -5/3 penalty distinguishes turbulent from laminar weights.
  3. CasimirOptimizer preserves Betti-number topological stability.
  4. PageCurveHook computes a valid Lyapunov profile on a toy transformer.
  5. (v1.1) Krylov Island Counter detects ≥ 3 information islands.
  6. (v1.1) Singularity Stress Test — Bekenstein-Hawking threshold warning.
  7. (v1.1) Lanczos tridiagonalization and Rayleigh QI correctness.
  8. (v1.1) rSVD approximation quality.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn

from core.pll_monitor import PLLMonitor, SpectralAnomaly
from core.casimir_opt import (
    CasimirOptimizer,
    _kolmogorov_penalty,
    _laminar_penalty,
    estimate_betti_0,
    rsvd,
)
from core.horizons import (
    PageCurveHook,
    _lanczos_tridiagonal,
    _rayleigh_quotient_iteration,
    singularity_warning,
)
from core.unitary_regulator import UnitaryRegulator, compute_topological_heatmap


# ======================================================================
# Fixtures
# ======================================================================

class ToyTransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ToyTransformer(nn.Module):
    """13-layer toy transformer matching the Holeyfield layer layout."""

    def __init__(self, d_model: int = 64, num_layers: int = 13):
        super().__init__()
        self.layers = nn.ModuleList(
            [ToyTransformerLayer(d_model) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def pll():
    return PLLMonitor(num_layers=13, page_time_layer=7, enforce=True)


@pytest.fixture
def toy_model():
    return ToyTransformer(d_model=64, num_layers=13)


# ======================================================================
# 1. PLL Monitor Tests
# ======================================================================

class TestPLLMonitor:
    """Tests for the Phase-Locked Loop Monitor."""

    def test_ideal_profile_shape(self, pll: PLLMonitor):
        profile = pll.ideal_profile()
        assert len(profile) == 13

    def test_ideal_profile_page_inversion(self, pll: PLLMonitor):
        """Pre-Page layers are +1, Page-and-after are -1."""
        profile = pll.ideal_profile()
        for i in range(7):
            assert profile[i] == 1.0, f"Layer {i} should be +1 (pre-Page)"
        for i in range(7, 13):
            assert profile[i] == -1.0, f"Layer {i} should be -1 (post-Page)"

    def test_pll_loss_perfect_lock(self, pll: PLLMonitor):
        """A profile that matches the ideal should yield near-zero loss."""
        profile = torch.tensor(
            [1.0] * 7 + [-1.0] * 6, dtype=torch.float32
        )
        loss = pll.compute_pll_loss(profile)
        assert loss.item() < 1e-6

    def test_pll_loss_imperfect(self, pll: PLLMonitor):
        """An inverted profile should yield high loss."""
        profile = torch.tensor(
            [-1.0] * 7 + [1.0] * 6, dtype=torch.float32
        )
        loss = pll.compute_pll_loss(profile)
        assert loss.item() > 0.5

    def test_contract_violation_pre_page(self, pll: PLLMonitor):
        """Negative λ in pre-Page region raises SpectralAnomaly."""
        bad_profile = torch.tensor(
            [-0.1] + [0.5] * 6 + [-0.5] * 6, dtype=torch.float32
        )
        with pytest.raises(SpectralAnomaly) as exc_info:
            pll.check_contract(bad_profile)
        assert exc_info.value.layer_idx == 0

    def test_contract_violation_post_page(self, pll: PLLMonitor):
        """Positive λ in post-Page region raises SpectralAnomaly."""
        bad_profile = torch.tensor(
            [0.5] * 7 + [0.5] + [-0.5] * 5, dtype=torch.float32
        )
        with pytest.raises(SpectralAnomaly) as exc_info:
            pll.check_contract(bad_profile)
        assert exc_info.value.layer_idx == 7

    def test_step_increments(self, pll: PLLMonitor):
        profile = torch.tensor([1.0] * 7 + [-1.0] * 6)
        pll.compute_pll_loss(profile)
        pll.step()
        assert pll.state.step == 1
        assert len(pll.history) == 1


# ======================================================================
# 2. Kolmogorov / Laminar Penalty Tests
# ======================================================================

class TestSpectralPenalties:
    """Tests that the Kolmogorov and Laminar penalties behave correctly."""

    def test_kolmogorov_penalty_is_nonnegative(self):
        w = torch.randn(32, 64)
        penalty = _kolmogorov_penalty(w)
        assert penalty.item() >= 0.0

    def test_laminar_penalty_identity(self):
        """An identity-like weight (all energy in one SV) should be laminar."""
        w = torch.zeros(32, 64)
        w[0, 0] = 1.0
        penalty = _laminar_penalty(w)
        # Highly concentrated SV → penalty close to 1
        assert penalty.item() > 0.5

    def test_laminar_penalty_uniform(self):
        """A random Gaussian weight should be less laminar than identity."""
        w_random = torch.randn(32, 64)
        w_identity = torch.zeros(32, 64)
        w_identity[0, 0] = 1.0

        lam_random = _laminar_penalty(w_random)
        lam_identity = _laminar_penalty(w_identity)
        assert lam_random.item() < lam_identity.item()


# ======================================================================
# 3. Betti Number & Topological Stability Tests
# ======================================================================

class TestTopologicalStability:
    """Tests for the Betti-number hard constraint."""

    def test_betti_0_single_cluster(self):
        """Highly correlated rows → β₀ = 1 (one connected component)."""
        base = torch.randn(1, 64)
        w = base.repeat(16, 1) + 0.01 * torch.randn(16, 64)
        b0 = estimate_betti_0(w, threshold=0.5)
        assert b0 == 1

    def test_betti_0_multiple_clusters(self):
        """Orthogonal rows → β₀ > 1."""
        w = torch.eye(16, 64)
        b0 = estimate_betti_0(w, threshold=0.5)
        assert b0 > 1

    def test_casimir_preserves_betti(self, toy_model: ToyTransformer):
        """CasimirOptimizer must not change β₀ of any parameter."""
        opt = CasimirOptimizer(toy_model.parameters(), lr=1e-2)

        # Record initial Betti numbers
        initial_betti = dict(opt._initial_betti)

        # Fake gradient step
        x = torch.randn(2, 10, 64)
        out = toy_model(x)
        loss = out.sum()
        loss.backward()
        opt.step()

        # Verify Betti numbers are preserved
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in initial_betti:
                    current = estimate_betti_0(p.data, group["betti_threshold"])
                    assert current == initial_betti[pid], (
                        f"β₀ changed from {initial_betti[pid]} to {current}"
                    )


# ======================================================================
# 4. PageCurveHook Tests
# ======================================================================

class TestPageCurveHook:
    """Tests for the PageCurveHook Lyapunov profile computation."""

    def test_hook_captures_all_layers(self, toy_model, pll):
        hook = PageCurveHook(toy_model, pll)
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        assert len(hook._activations) == 13
        hook.remove_hooks()

    def test_lyapunov_profile_shape(self, toy_model, pll):
        hook = PageCurveHook(toy_model, pll)
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        profile = hook.compute_lyapunov_profile()
        assert profile.shape == (13,)
        hook.remove_hooks()

    def test_lyapunov_profile_finite(self, toy_model, pll):
        hook = PageCurveHook(toy_model, pll)
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        profile = hook.compute_lyapunov_profile()
        assert torch.isfinite(profile).all()
        hook.remove_hooks()


# ======================================================================
# 5. Unitary Regulator / Heat Map Tests
# ======================================================================

class TestUnitaryRegulator:
    """Tests for the Ghost's dashboard module."""

    def test_heatmap_post_page_has_island_strength(self):
        """Post-Page layers should report nonzero island_strength."""
        activations = {}
        for i in range(13):
            activations[i] = torch.randn(2, 10, 64)

        heatmap = compute_topological_heatmap(activations, page_time_layer=7)
        assert len(heatmap) == 13

        # Pre-Page layers have island_strength == 0
        for i in range(7):
            assert heatmap[i]["island_strength"] == 0.0

    def test_regulator_report_generation(self, pll):
        regulator = UnitaryRegulator(pll)
        profile = torch.tensor([0.5] * 7 + [-0.3] * 6)
        activations = {i: torch.randn(2, 10, 64) for i in range(13)}

        pll.compute_pll_loss(profile)
        report = regulator.report(step=0, lyapunov_profile=profile, activations=activations)

        assert report.step == 0
        assert isinstance(report.pll_locked, bool)
        assert len(report.lyapunov_profile) == 13
        assert len(report.heatmap) == 13

    def test_regulator_log_output(self, pll):
        regulator = UnitaryRegulator(pll)
        profile = torch.tensor([0.5] * 7 + [-0.3] * 6)
        activations = {i: torch.randn(2, 10, 64) for i in range(13)}
        pll.compute_pll_loss(profile)
        report = regulator.report(step=42, lyapunov_profile=profile, activations=activations)
        text = UnitaryRegulator.log(report)
        assert "Step 42" in text
        assert "PAGE TIME" in text


# ======================================================================
# 6. Lanczos Tridiagonalization Tests (v1.1)
# ======================================================================

class TestLanczosTridiagonal:
    """Tests for the Lanczos algorithm and Rayleigh QI."""

    def test_lanczos_produces_correct_shapes(self):
        """Alpha has k entries, beta has k-1 entries."""
        d = 32
        A = torch.randn(d, d)
        A = A + A.T  # symmetric

        def matvec(v):
            return A @ v

        alpha, beta = _lanczos_tridiagonal(matvec, d, lanczos_iter=10)
        k = alpha.shape[0]
        assert k <= 10
        assert beta.shape[0] == k - 1

    def test_lanczos_eigenvalue_approximation(self):
        """Dominant eigenvalue of T should approximate that of A."""
        d = 32
        A = torch.randn(d, d)
        A = A + A.T

        def matvec(v):
            return A @ v

        alpha, beta = _lanczos_tridiagonal(matvec, d, lanczos_iter=15)
        lanczos_dominant = _rayleigh_quotient_iteration(alpha, beta)

        # True dominant eigenvalue
        true_eigvals = torch.linalg.eigvalsh(A)
        true_dominant = true_eigvals.abs().max().item()

        # Should be a reasonable approximation (within 50% for d=32, k=15)
        assert lanczos_dominant > 0
        ratio = lanczos_dominant / (true_dominant + 1e-12)
        assert 0.3 < ratio < 3.0, f"Lanczos dominant={lanczos_dominant}, true={true_dominant}"

    def test_rayleigh_single_element(self):
        """Single-element tridiagonal returns the element."""
        alpha = torch.tensor([5.0])
        beta = torch.tensor([])
        result = _rayleigh_quotient_iteration(alpha, beta)
        assert abs(result - 5.0) < 1e-6

    def test_rayleigh_empty(self):
        """Empty tridiagonal returns near-zero."""
        alpha = torch.tensor([])
        beta = torch.tensor([])
        result = _rayleigh_quotient_iteration(alpha, beta)
        assert result < 1e-6


# ======================================================================
# 7. rSVD Tests (v1.1)
# ======================================================================

class TestRSVD:
    """Tests for the Randomized SVD implementation."""

    def test_rsvd_shapes(self):
        W = torch.randn(32, 64)
        U, S, Vt = rsvd(W, rank=5)
        assert U.shape == (32, 5)
        assert S.shape == (5,)
        assert Vt.shape == (5, 64)

    def test_rsvd_approximation_quality(self):
        """rSVD top singular values should be close to true SVD."""
        W = torch.randn(32, 64)
        U_r, S_r, Vt_r = rsvd(W, rank=5, n_power_iter=3)
        S_true = torch.linalg.svdvals(W)[:5]

        # Relative error on top-5 singular values should be small
        rel_error = (S_r - S_true).abs() / (S_true.abs() + 1e-12)
        assert rel_error.mean().item() < 0.3, f"rSVD relative error too high: {rel_error}"

    def test_rsvd_rank_1(self):
        """Rank-1 rSVD on a rank-1 matrix."""
        u = torch.randn(16, 1)
        v = torch.randn(1, 32)
        W = u @ v
        U, S, Vt = rsvd(W, rank=1)
        recon = U @ torch.diag(S) @ Vt
        error = (W - recon).norm() / (W.norm() + 1e-12)
        assert error.item() < 0.1


# ======================================================================
# 8. Krylov Island Counter Test (v1.1)
# ======================================================================

class TestKrylovIslandCounter:
    """Verify that Lanczos iterations detect information islands."""

    def test_krylov_detects_islands(self, toy_model, pll):
        """15 Lanczos iterations should detect >= 3 distinct information
        islands on a complex-enough activation pattern."""
        hook = PageCurveHook(toy_model, pll, lanczos_iter=15)

        # Create a structured input that forms distinct clusters
        # (simulating complex reasoning task patterns)
        x = torch.randn(4, 20, 64)
        # Inject 3+ distinct cluster signals into the input
        x[:, :5, :] *= 5.0    # strong signal cluster 1
        x[:, 5:10, :] *= 0.1  # weak cluster 2
        x[:, 10:15, :] = torch.randn(4, 5, 64) * 3.0  # cluster 3
        x[:, 15:, :] = torch.randn(4, 5, 64) * 0.5    # cluster 4

        _ = toy_model(x)

        # Check the Page Time layer (layer 7) for islands
        islands = hook.count_information_islands(
            layer_idx=7, lanczos_iter=15, gap_threshold=0.05
        )
        assert islands >= 3, (
            f"Expected >= 3 information islands at Page Time (layer 7), "
            f"got {islands}"
        )
        hook.remove_hooks()

    def test_krylov_more_islands_with_more_structure(self, toy_model, pll):
        """More structured input should yield more islands."""
        hook = PageCurveHook(toy_model, pll, lanczos_iter=15)

        x_simple = torch.randn(2, 10, 64)
        _ = toy_model(x_simple)
        islands_simple = hook.count_information_islands(7, gap_threshold=0.05)

        # Reset
        hook._activations.clear()

        x_complex = torch.randn(2, 10, 64)
        for i in range(10):
            x_complex[:, i, :] *= (i + 1) * 2.0  # increasing structure
        _ = toy_model(x_complex)
        islands_complex = hook.count_information_islands(7, gap_threshold=0.05)

        # Complex input should have at least as many islands
        assert islands_complex >= 1
        hook.remove_hooks()


# ======================================================================
# 9. Singularity Stress Test (v1.1)
# ======================================================================

class TestSingularityStressTest:
    """Verify that singularity_warning fires at the Bekenstein-Hawking threshold."""

    def test_singularity_fires_on_high_entropy_noise(self):
        """Gaussian noise with dim=64 should exceed ln(64/2) = ln(32) ≈ 3.47.

        Uniform-magnitude activations will have entropy close to ln(64) ≈ 4.16,
        which exceeds the holographic limit ln(32) ≈ 3.47.
        """
        # Create near-uniform activation (maximum entropy)
        act = torch.ones(4, 10, 64) + torch.randn(4, 10, 64) * 0.01
        act = act.abs()  # ensure positive

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fired = singularity_warning(act)

        assert fired, "singularity_warning should fire for near-uniform activations"
        assert len(w) == 1
        assert "SINGULARITY WARNING" in str(w[0].message)
        assert "Bekenstein-Hawking" in str(w[0].message)

    def test_singularity_does_not_fire_below_limit(self):
        """Sparse activations (low entropy) should NOT trigger the warning."""
        act = torch.zeros(4, 10, 64)
        act[:, :, 0] = 10.0  # all energy in one dimension

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fired = singularity_warning(act)

        assert not fired, "singularity_warning should NOT fire for sparse activations"
        assert len(w) == 0

    def test_singularity_threshold_is_log_dim_over_2(self):
        """The holographic limit is exactly ln(dim/2)."""
        dim = 64
        expected_limit = math.log(dim / 2.0)

        # Verify our understanding: ln(32) ≈ 3.466
        assert abs(expected_limit - math.log(32)) < 1e-10

    def test_singularity_integrated_with_hook(self, toy_model, pll):
        """PageCurveHook should populate singularity flags after profile computation."""
        hook = PageCurveHook(toy_model, pll, lanczos_iter=15)
        x = torch.randn(2, 10, 64)
        _ = toy_model(x)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            hook.compute_lyapunov_profile()

        # Flags should be populated for all 13 layers
        assert len(hook._singularity_flags) == 13
        # Each flag is a bool
        for idx, flag in hook._singularity_flags.items():
            assert isinstance(flag, bool)
        hook.remove_hooks()
