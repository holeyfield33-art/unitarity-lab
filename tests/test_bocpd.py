"""
tests/test_bocpd.py — Validated tests for the two-model log-space BOCPD detector.

Pure CPU, seeded numpy only.  No torch or model weights required.

Test coverage
-------------
1. test_synthetic_regime_change_detected  — logic floor: detect GUE→collapse
2. test_stable_stream_no_false_alarm      — no false positives on stable data
3. test_calibration_sets_baseline_from_warmup — measured-baseline calibration
4. test_warmup_suppresses_alarms          — warm-up returns exact 0.0
5. test_return_range                      — every return is in [0.0, 1.0]
6. test_real_baseline_detection           — SKIPPED unless fixture file present
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from unitarity_labs.core.bocpd import PredictiveAnomalyDetector

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "real_r_trace_collapse.npy"
fixture_exists = FIXTURE_PATH.exists()


# ---------------------------------------------------------------------------
# 1. Synthetic regime-change detection (logic floor)
# ---------------------------------------------------------------------------

def test_synthetic_regime_change_detected():
    """Validated scenario: ~950 stable GUE samples then ~769 collapsed samples.

    Detection must fire and the detected index must be within 150 steps of the
    true changepoint at index 950.
    """
    rng = np.random.default_rng(42)
    stable = rng.normal(0.638, 0.015, 950)
    collapsed = rng.normal(0.42, 0.015, 769)
    r_sequence = np.concatenate([stable, collapsed])

    # mean_0 provided explicitly → no warmup, uses validated params
    det = PredictiveAnomalyDetector(
        mean_0=0.638,
        mean_1=0.42,
        hazard_rate=1000.0,
        threshold=0.95,
    )

    detected_index = None
    for i, r in enumerate(r_sequence):
        # zeta held constant; r_ratio drives detection (raw r-ratio units)
        prob = det.process_step(zeta=1.0, r_ratio=float(r))
        if prob > det.threshold and detected_index is None:
            detected_index = i

    assert detected_index is not None, "Detector never fired on a synthetic collapse"
    assert abs(detected_index - 950) < 150, (
        f"Detection at {detected_index} is more than 150 steps from true changepoint 950"
    )


# ---------------------------------------------------------------------------
# 2. Stable stream — no false alarms
# ---------------------------------------------------------------------------

def test_stable_stream_no_false_alarm():
    """1000 stable GUE samples must produce zero alarms (P < threshold every step)."""
    rng = np.random.default_rng(42)
    stable = rng.normal(0.638, 0.015, 1000)

    det = PredictiveAnomalyDetector(
        mean_0=0.638,
        mean_1=0.42,
        hazard_rate=1000.0,
        threshold=0.95,
    )

    for i, r in enumerate(stable):
        prob = det.process_step(zeta=1.0, r_ratio=float(r))
        assert prob < det.threshold, (
            f"False alarm at step {i}: prob={prob:.6f} ≥ threshold={det.threshold}"
        )


# ---------------------------------------------------------------------------
# 3. Calibration from warm-up
# ---------------------------------------------------------------------------

def test_calibration_sets_baseline_from_warmup():
    """mean_0=None + warmup_steps=100: after 100 samples the detector self-calibrates."""
    rng = np.random.default_rng(7)
    samples = rng.normal(0.638, 0.015, 100)

    det = PredictiveAnomalyDetector(mean_0=None, warmup_steps=100)

    for r in samples:
        det.process_step(zeta=1.0, r_ratio=float(r))

    d = det.diagnostics()
    print(f"\nCalibrated mean_0: {d['mean_0']:.6f}")   # reported in verification

    assert d["calibrated"] is True, "diagnostics()['calibrated'] should be True after warm-up"
    assert abs(d["mean_0"] - 0.638) < 0.01, (
        f"Calibrated mean_0={d['mean_0']:.6f} deviates more than 0.01 from 0.638"
    )


# ---------------------------------------------------------------------------
# 4. Warm-up suppresses alarms
# ---------------------------------------------------------------------------

def test_warmup_suppresses_alarms():
    """The first warmup_steps calls must return exactly 0.0 regardless of input."""
    det = PredictiveAnomalyDetector(mean_0=None, warmup_steps=50)

    # Feed "wild" values that would otherwise trigger alarms
    wild_values = [0.0, 1e6, -1e6, 0.0, 999.9, 0.42, 0.638] + [0.1] * 43

    for i, v in enumerate(wild_values):
        result = det.process_step(zeta=1.0, r_ratio=float(v))
        assert result == 0.0, (
            f"Step {i}: expected 0.0 during warm-up, got {result}"
        )


# ---------------------------------------------------------------------------
# 5. Return range
# ---------------------------------------------------------------------------

def test_return_range():
    """Every process_step return must be in [0.0, 1.0]."""
    rng = np.random.default_rng(99)
    det = PredictiveAnomalyDetector(mean_0=0.638, mean_1=0.42, hazard_rate=1000.0)

    # Mix of stable, collapsed, and extreme values
    values = np.concatenate([
        rng.normal(0.638, 0.015, 200),
        rng.normal(0.42, 0.015, 200),
        np.array([0.0, 1.0, -1.0, 1e3, -1e3]),
    ])

    for i, v in enumerate(values):
        prob = det.process_step(zeta=1.0, r_ratio=float(v))
        assert 0.0 <= prob <= 1.0, (
            f"Step {i}: out-of-range return {prob} for r_ratio={v}"
        )


# ---------------------------------------------------------------------------
# 6. Real-baseline detection (skipped unless fixture file is present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not fixture_exists,
    reason="Drop tests/fixtures/real_r_trace_collapse.npy to enable this test",
)
def test_real_baseline_detection():
    """Asserts calibration + detection on a real recorded r-ratio trace.

    The fixture must be a 1-D float64 .npy array where the first portion
    represents a stable GUE run and the tail contains a real collapse.
    Expected: self-calibration succeeds and an alarm fires in the collapse tail.
    """
    r_trace = np.load(FIXTURE_PATH).astype(np.float64)
    assert r_trace.ndim == 1 and len(r_trace) >= 200, (
        "Fixture must be a 1-D array with at least 200 samples"
    )

    det = PredictiveAnomalyDetector(
        mean_0=None,          # self-calibrate from warm-up
        warmup_steps=100,
        hazard_rate=1000.0,
        threshold=0.95,
    )

    detected = False
    for r in r_trace:
        prob = det.process_step(zeta=1.0, r_ratio=float(r))
        if prob > det.threshold:
            detected = True
            break

    d = det.diagnostics()
    assert d["calibrated"] is True, "Detector did not complete calibration"
    assert detected, "No changepoint detected in real collapse trace"
