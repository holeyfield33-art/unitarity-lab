"""
test_passive_hook.py — PassiveTelemetryHook regression tests.
================================================================
Covers Phase 2 of the insideai/unitarity-lab/VAR consolidation plan:
a clean passive-mode hook returning {zeta_raw, spectral_gap, flagged},
with flagged driven by VAR's real SpectralRuptureDetector rather than a
hardcoded value, and zero impact on the passive-mode no-mutation
guarantee already covered by test_hardening.py.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import ToyTransformer
from var.detector import SpectralRuptureDetector


class _ToyConfig:
    hidden_size = 64
    num_attention_heads = 8


def _make_passive_wrapper():
    from unitarity_labs.core.universal_hook import UniversalHookWrapper

    model = ToyTransformer(d_model=64, num_layers=13)
    return UniversalHookWrapper(
        model=model, config=_ToyConfig(), node_id="A", mode="passive",
    )


class TestPassiveTelemetryHook:
    def test_requires_passive_mode(self):
        from unitarity_labs.core.passive_hook import PassiveTelemetryHook
        from unitarity_labs.core.universal_hook import UniversalHookWrapper

        model = ToyTransformer(d_model=64, num_layers=13)
        active_wrapper = UniversalHookWrapper(
            model=model, config=_ToyConfig(), node_id="A", mode="active",
        )
        with pytest.raises(ValueError):
            PassiveTelemetryHook(active_wrapper)

    def test_read_returns_exactly_three_fields(self):
        from unitarity_labs.core.passive_hook import PassiveTelemetryHook

        wrapper = _make_passive_wrapper()
        hook = PassiveTelemetryHook(wrapper)
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            wrapper(x)
        telemetry = hook.read()
        assert set(telemetry.keys()) == {"zeta_raw", "spectral_gap", "flagged"}
        assert isinstance(telemetry["zeta_raw"], float)
        assert isinstance(telemetry["spectral_gap"], float)
        assert isinstance(telemetry["flagged"], bool)

    def test_flagged_uses_real_rupture_detector_not_hardcoded(self):
        """flagged must come from the detector's real calibrated threshold
        check, not a constant: a detector calibrated on a baseline the
        observed spectral_gap clearly exceeds must fire True, and a detector
        calibrated around that same spectral_gap value must stay False —
        both driven by the same wrapper output, only the calibration differs."""
        from unitarity_labs.core.passive_hook import PassiveTelemetryHook

        wrapper = _make_passive_wrapper()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            wrapper(x)
        observed_gap = float(wrapper.get_metrics()["spectral_gap"])

        rng = __import__("random").Random(0)

        # Calibrated far below the observed gap, tight spread -> threshold is
        # comfortably crossed by the very next sample (hysteresis=1 fires).
        elevated_detector = SpectralRuptureDetector(warmup=0, calib_len=20, hysteresis=1)
        for _ in range(20):
            elevated_detector.update(observed_gap - 1.0 + rng.uniform(-0.001, 0.001))
        hook_elevated = PassiveTelemetryHook(wrapper, rupture_detector=elevated_detector)
        assert hook_elevated.read()["flagged"] is True

        # Calibrated around the observed gap itself with a wide spread -> the
        # same sample falls well inside the healthy band.
        calm_detector = SpectralRuptureDetector(warmup=0, calib_len=20, hysteresis=1)
        for _ in range(20):
            calm_detector.update(observed_gap + rng.uniform(-0.5, 0.5))
        hook_calm = PassiveTelemetryHook(wrapper, rupture_detector=calm_detector)
        assert hook_calm.read()["flagged"] is False

    def test_read_does_not_mutate_wrapper_output(self):
        """Reading telemetry must not change the passive-mode no-mutation
        guarantee: wrapped output stays byte-identical to the hookless model."""
        from unitarity_labs.core.passive_hook import PassiveTelemetryHook
        from unitarity_labs.core.universal_hook import UniversalHookWrapper

        x = torch.randn(1, 8, 64)

        torch.manual_seed(7)
        m_bare = ToyTransformer(d_model=64, num_layers=13)
        with torch.no_grad():
            out_bare = m_bare(x)

        torch.manual_seed(7)
        m_pass = ToyTransformer(d_model=64, num_layers=13)
        wrapper = UniversalHookWrapper(model=m_pass, config=_ToyConfig(), mode="passive")
        hook = PassiveTelemetryHook(wrapper)
        with torch.no_grad():
            out_pass = wrapper(x)
        hook.read()

        assert torch.equal(out_bare, out_pass), \
            "PassiveTelemetryHook must not affect passive-mode byte-identical output"
