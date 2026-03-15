"""
test_hardening.py — v3.1.1-Singularity TMRP-13 Regression Tests
=================================================================
Covers all 10 phases of the hardening pass:

  1. Version consistency
  2. Three-tier imports (core / dist / labs)
  3. Manifold Coherence ζ correctness
  4. Passive mode no-mutation guarantee
  5. Active mode still intervenes
  6. ChronosLock isolation from single-node
  7. Tier admission / demotion logic
  8. Router quorum bypass
  9. Diversity snapshot trigger
 10. Benchmark harness boot
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.conftest import ToyTransformer


# ======================================================================
# Helpers
# ======================================================================

class _ToyConfig:
    """Minimal HF-style config for UniversalHookWrapper."""
    hidden_size = 64
    num_attention_heads = 8


def _make_wrapper(mode: str = "active", enable_dual: bool = False):
    """Build a UniversalHookWrapper around a ToyTransformer."""
    from core.universal_hook import UniversalHookWrapper

    model = ToyTransformer(d_model=64, num_layers=13)
    return UniversalHookWrapper(
        model=model,
        config=_ToyConfig(),
        node_id="A",
        enable_dual=enable_dual,
        mode=mode,
    )


# ======================================================================
# Phase 1 — Version Consistency
# ======================================================================

class TestVersionConsistency:
    def test_core_version_is_singularity(self):
        from core.version import __version__
        assert __version__ == "3.1.1-Singularity"

    def test_init_reexports_version(self):
        import core
        assert hasattr(core, "__version__")
        assert core.__version__ == "3.1.1-Singularity"

    def test_setup_py_version_matches(self):
        """setup.py must declare the same version string."""
        import ast
        from pathlib import Path

        setup_src = Path("setup.py").read_text(encoding="utf-8")
        tree = ast.parse(setup_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "version":
                if isinstance(node.value, ast.Constant):
                    assert node.value.value == "3.1.1-Singularity"
                    return
        pytest.fail("Could not find version= in setup.py")


# ======================================================================
# Phase 2 — Three-Tier Imports
# ======================================================================

class TestThreeTierImports:
    def test_core_import(self):
        import core
        assert hasattr(core, "UniversalHookWrapper")
        assert hasattr(core, "manifold_coherence_zeta")

    def test_dist_import(self):
        import dist
        assert hasattr(dist, "__version__")

    def test_dist_tier_manager(self):
        from dist.tier_manager import TierManager, NodeTier
        tm = TierManager()
        assert tm is not None
        assert NodeTier.COMPUTE.value == "COMPUTE"
        assert NodeTier.ROUTER.value == "ROUTER"

    def test_labs_import(self):
        import labs
        # labs is experimental — just check it imports
        assert labs is not None

    def test_labs_topology_metrics(self):
        from labs.topology_metrics import spectral_gap_from_activations
        act = {0: torch.randn(1, 16, 64), 1: torch.randn(1, 16, 64)}
        result = spectral_gap_from_activations(act)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)


# ======================================================================
# Phase 3 — Manifold Coherence ζ
# ======================================================================

class TestZetaMetric:
    def test_identical_tensors_yield_zeta_one(self):
        from core.metrics import manifold_coherence_zeta
        t = torch.randn(2, 32, 64)
        z = manifold_coherence_zeta(t, t)
        assert abs(z - 1.0) < 1e-5

    def test_orthogonal_tensors_yield_low_zeta(self):
        from core.metrics import manifold_coherence_zeta
        a = torch.zeros(1, 4, 8)
        a[0, 0, 0] = 1.0
        b = torch.zeros(1, 4, 8)
        b[0, 1, 1] = 1.0
        z = manifold_coherence_zeta(a, b)
        assert abs(z) < 1e-5

    def test_baseline_cosine_range(self):
        from core.metrics import baseline_cosine_meanpool
        a = torch.randn(1, 16, 64)
        b = torch.randn(1, 16, 64)
        c = baseline_cosine_meanpool(a, b)
        assert -1.0 <= c <= 1.0

    def test_permutation_test_reproducibility(self):
        from core.metrics import permutation_test_zeta
        a = torch.randn(1, 16, 64)
        b = a + 0.01 * torch.randn(1, 16, 64)
        z1, p1 = permutation_test_zeta(a, b, n_perm=50, seed=123)
        z2, p2 = permutation_test_zeta(a, b, n_perm=50, seed=123)
        assert z1 == z2
        assert p1 == p2


# ======================================================================
# Phase 4 — Passive Mode No-Mutation Guarantee
# ======================================================================

class TestPassiveMode:
    def test_passive_wrapper_creates(self):
        w = _make_wrapper(mode="passive")
        assert w.mode == "passive"
        assert w.bridge.enabled is False

    def test_passive_no_tensor_mutation(self):
        """In passive mode, model output must equal a bare model's output."""
        torch.manual_seed(42)
        model_bare = ToyTransformer(d_model=64, num_layers=13)

        torch.manual_seed(42)
        model_wrapped = ToyTransformer(d_model=64, num_layers=13)
        w = _make_wrapper.__wrapped__(model_wrapped) if hasattr(_make_wrapper, '__wrapped__') else None

        # Build wrapper directly to reuse same model
        from core.universal_hook import UniversalHookWrapper
        torch.manual_seed(42)
        m = ToyTransformer(d_model=64, num_layers=13)
        wrapper = UniversalHookWrapper(
            model=m, config=_ToyConfig(), mode="passive",
        )

        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            # Re-create bare model with same seed
            torch.manual_seed(42)
            m_bare = ToyTransformer(d_model=64, num_layers=13)
            out_bare = m_bare(x)

            torch.manual_seed(42)
            m_pass = ToyTransformer(d_model=64, num_layers=13)
            wp = UniversalHookWrapper(
                model=m_pass, config=_ToyConfig(), mode="passive",
            )
            out_pass = wp(x)

        assert torch.allclose(out_bare, out_pass, atol=1e-6), \
            "Passive mode must not mutate tensors"

    def test_passive_metrics_include_mode(self):
        w = _make_wrapper(mode="passive")
        m = w.get_metrics()
        assert m["mode"] == "passive"

    def test_passive_metrics_include_zeta(self):
        w = _make_wrapper(mode="passive")
        m = w.get_metrics()
        assert "manifold_coherence_zeta" in m


# ======================================================================
# Phase 5 — Active Mode Still Intervenes
# ======================================================================

class TestActiveMode:
    def test_active_wrapper_creates(self):
        w = _make_wrapper(mode="active")
        assert w.mode == "active"
        assert w.bridge.enabled is True

    def test_active_metrics_include_mode(self):
        w = _make_wrapper(mode="active")
        m = w.get_metrics()
        assert m["mode"] == "active"

    def test_invalid_mode_raises(self):
        from core.universal_hook import UniversalHookWrapper
        model = ToyTransformer(d_model=64, num_layers=13)
        with pytest.raises(ValueError, match="mode must be"):
            UniversalHookWrapper(model=model, config=_ToyConfig(), mode="invalid")


# ======================================================================
# Phase 6 — ChronosLock Isolation
# ======================================================================

class TestChronosLockIsolation:
    def test_single_node_does_not_use_chronos(self):
        """UniversalHookWrapper with enable_dual=False must not reference ChronosLock."""
        w = _make_wrapper(mode="active", enable_dual=False)
        # The wrapper itself has no chronos_lock attribute
        assert not hasattr(w, "chronos_lock")

    def test_dist_chronos_imports(self):
        """dist.chronos_lock must be importable."""
        from dist.chronos_lock import ChronosLock
        assert ChronosLock is not None


# ======================================================================
# Phase 7 — Tier Admission / Demotion
# ======================================================================

class TestTierPolicing:
    def test_attest_compute(self):
        from dist.tier_manager import TierManager, NodeTier
        tm = TierManager(min_compute_tps=10.0)
        tier = tm.attest("node-1", tps_ema=20.0, tps_variance=1.0)
        assert tier == NodeTier.COMPUTE

    def test_attest_router(self):
        from dist.tier_manager import TierManager, NodeTier
        tm = TierManager(min_compute_tps=10.0)
        tier = tm.attest("node-1", tps_ema=5.0, tps_variance=0.5)
        assert tier == NodeTier.ROUTER

    def test_demotion_on_wait(self):
        from dist.tier_manager import TierManager, NodeTier
        tm = TierManager(min_compute_tps=10.0)
        tm.attest("node-1", tps_ema=20.0, tps_variance=1.0)
        # Simulate excessive wait that triggers demotion
        tm.record_wait("node-1", wait_secs=3.0)
        rec = tm.get_record("node-1")
        assert rec.tier == NodeTier.ROUTER

    def test_compute_quorum(self):
        from dist.tier_manager import TierManager
        tm = TierManager(min_compute_tps=10.0)
        tm.attest("a", tps_ema=20.0, tps_variance=1.0)
        tm.attest("b", tps_ema=20.0, tps_variance=1.0)
        tm.attest("c", tps_ema=5.0, tps_variance=0.5)
        # 2 compute + 1 router: quorum with max_faulty=0 (need 1 compute)
        assert tm.compute_quorum_met(max_faulty=0)
        # quorum with max_faulty=1 requires 3 compute
        assert not tm.compute_quorum_met(max_faulty=1)

    def test_router_quorum_bypass(self):
        """Router nodes should not count toward compute quorum."""
        from dist.tier_manager import TierManager
        tm = TierManager(min_compute_tps=10.0)
        tm.attest("r1", tps_ema=5.0, tps_variance=0.5)
        tm.attest("r2", tps_ema=5.0, tps_variance=0.5)
        # 0 compute nodes — quorum never met even with max_faulty=0
        assert not tm.compute_quorum_met(max_faulty=0)


# ======================================================================
# Phase 8 — Diversity Snapshot Trigger
# ======================================================================

class TestDiversitySnapshot:
    def test_snapshot_monitor_creates(self):
        from core.diversity_snapshot import DiversitySnapshotMonitor
        mon = DiversitySnapshotMonitor()
        assert mon is not None

    def test_should_disable_bridge_during_solo(self):
        from core.diversity_snapshot import DiversitySnapshotMonitor
        mon = DiversitySnapshotMonitor()
        # Advance to snapshot interval
        for _ in range(4096):
            mon.step()
        assert mon.should_disable_bridge is True

    def test_collapse_detection(self):
        from core.diversity_snapshot import DiversitySnapshotMonitor
        mon = DiversitySnapshotMonitor()

        h = torch.randn(1, 16, 64)

        # First checkpoint: advance to snapshot boundary, enter solo window
        for _ in range(4096):
            mon.step()
        assert mon.in_solo_window
        # Record nearly-identical states during the solo window
        for _ in range(128):
            mon.record_states(h_solo=h, h_bridged=h + 1e-8 * torch.randn_like(h))
            mon.step()
        # Solo window should have finalized
        assert not mon.in_solo_window

        # Second checkpoint
        for _ in range(4096 - 128):
            mon.step()
        # Compensate: next interval triggers at next 4096 boundary
        # We may need more steps. Let's just step until in solo again.
        while not mon.in_solo_window:
            mon.step()
        for _ in range(128):
            mon.record_states(h_solo=h, h_bridged=h + 1e-8 * torch.randn_like(h))
            mon.step()

        # After 2 consecutive low-delta checkpoints, collapse warning count should increase
        assert mon.collapse_warning_count >= 1


# ======================================================================
# Phase 9 — Benchmark Harness Boot
# ======================================================================

class TestBenchmarkHarness:
    def test_harness_imports(self):
        from benchmarks._harness import make_parser, set_seed, compute_row, emit
        assert callable(make_parser)
        assert callable(compute_row)

    def test_compute_row_columns(self):
        from benchmarks._harness import compute_row
        source = torch.randn(1, 16, 64)
        sink = source + 0.05 * torch.randn(1, 16, 64)
        row = compute_row(source, sink, latency_ms=5.0, accuracy=0.9)
        assert set(row.keys()) == {"zeta", "baseline_cosine", "permutation_p", "latency_ms", "accuracy"}

    def test_gsm8k_runs(self):
        """gsm8k benchmark must run without error."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.gsm8k", "--n-problems", "2", "--seed", "1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "results" in data
        assert len(data["results"]) == 2


# ======================================================================
# Phase 10 — Smoke Tests
# ======================================================================

class TestSmokeTests:
    def test_core_only_import(self):
        """Importing core alone must not raise."""
        import core  # noqa: F401

    def test_dist_import(self):
        import dist  # noqa: F401

    def test_labs_import(self):
        import labs  # noqa: F401
