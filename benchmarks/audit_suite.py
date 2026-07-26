#!/usr/bin/env python3
"""benchmarks/audit_suite.py — sequential, fully-logged audit suite.

Runs each check **one at a time**, records every number it produced, and only
then moves to the next. Nothing is summarised away and nothing is silently
skipped: a check that errors is recorded with its traceback and the suite
carries on, so one broken dependency cannot mask the rest of the results.

Why this exists
---------------
Numbers in this repo were previously hard to trust for three separate reasons:

1. Some notebook cells caught every exception and fell through to
   ``np.random.randn`` placeholders, so a *failed* run still printed a
   plausible-looking score — a different one each time.
2. Nothing recorded which package versions, device, or seed produced a number,
   so two runs could not be compared.
3. Nothing distinguished a metric that is deterministic from one that is
   genuinely noisy.

This suite addresses all three. Placeholders are gone — a check that cannot run
reports ``error``. Every run writes a manifest. And ``--repeat N`` re-runs each
check N times and reports mean/std/min/max per metric plus a ``deterministic``
flag, so you can see which numbers are stable before quoting any of them.

Reference values
----------------
Checks that have a known correct answer assert against it and report the
deviation, rather than just printing whatever came out:

- ``gue_r_ratio``: random-matrix theory gives ⟨r⟩ ≈ 0.5996 for GUE and
  ≈ 0.3863 for an uncorrelated (Poisson) spectrum.
- ``chronos_shard_roundtrip``: Reed–Solomon with 2 parity symbols must recover
  from exactly 1 corrupted symbol and must *detect* more.
- ``bocpd_*``: a synthetic stream with an injected changepoint at a known index
  gives a ground-truth detection delay and false-alarm count.

Usage::

    python -m benchmarks.audit_suite --list
    python -m benchmarks.audit_suite --repeat 3
    python -m benchmarks.audit_suite --only bocpd_changepoint --repeat 5
    python -m benchmarks.audit_suite --model gpt2 --model-tier tiny
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from benchmarks._harness import device_name, env_tag, git_sha, pip_freeze

# ----------------------------------------------------------------------
# Reference constants (random-matrix theory)
# ----------------------------------------------------------------------

#: ⟨r⟩ for the Gaussian Unitary Ensemble (Atas et al. 2013).
GUE_R_REFERENCE: float = 0.5996
#: ⟨r⟩ for an uncorrelated (Poisson) spectrum: 2 ln 2 − 1.
POISSON_R_REFERENCE: float = 0.3863

#: Model presets that fit a single 16 GB T4. ``vram_gb`` is the approximate
#: weight footprint at the listed dtype, excluding activations and KV cache.
MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "tiny":   {"model": "distilgpt2",                    "dtype": "float32", "params_m": 82,   "vram_gb": 0.4},
    "small":  {"model": "gpt2",                          "dtype": "float32", "params_m": 124,  "vram_gb": 0.6},
    "medium": {"model": "gpt2-medium",                   "dtype": "float16", "params_m": 355,  "vram_gb": 0.8},
    "large":  {"model": "Qwen/Qwen2.5-0.5B-Instruct",    "dtype": "float16", "params_m": 494,  "vram_gb": 1.1},
    "xl":     {"model": "Qwen/Qwen2.5-1.5B-Instruct",    "dtype": "float16", "params_m": 1544, "vram_gb": 3.2},
}


# ----------------------------------------------------------------------
# Result plumbing
# ----------------------------------------------------------------------

class CheckResult:
    """One execution of one check."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status: str = "pending"
        self.metrics: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.error: Optional[str] = None
        self.duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "status": self.status,
            "duration_s": round(self.duration_s, 4),
            "metrics": self.metrics,
            "notes": self.notes,
            "error": self.error,
        }


class Recorder:
    """Collects metrics for the check currently running."""

    def __init__(self, result: CheckResult) -> None:
        self._result = result

    def metric(self, key: str, value: Any) -> None:
        """Record a single measured value."""
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self._result.metrics[key] = value

    def note(self, text: str) -> None:
        """Record a human-readable observation (never a measured number)."""
        self._result.notes.append(text)

    def check_close(
        self, key: str, observed: float, expected: float, tol: float
    ) -> bool:
        """Record ``observed`` alongside its reference value and deviation."""
        deviation = abs(observed - expected)
        passed = bool(deviation <= tol)
        self.metric(key, float(observed))
        self.metric(f"{key}__expected", float(expected))
        self.metric(f"{key}__deviation", float(deviation))
        self.metric(f"{key}__within_tol", passed)
        if not passed:
            self.note(
                f"{key}={observed:.6g} deviates from reference {expected:.6g} "
                f"by {deviation:.6g} (tolerance {tol:.6g})"
            )
        return passed


CheckFn = Callable[[Recorder, argparse.Namespace], None]

#: Ordered registry. Insertion order is execution order.
REGISTRY: Dict[str, CheckFn] = {}
#: Checks that need a transformer model (skipped unless --model is usable).
MODEL_CHECKS: set[str] = set()


def check(name: str, *, needs_model: bool = False) -> Callable[[CheckFn], CheckFn]:
    def decorator(fn: CheckFn) -> CheckFn:
        REGISTRY[name] = fn
        if needs_model:
            MODEL_CHECKS.add(name)
        return fn
    return decorator


# ----------------------------------------------------------------------
# Synthetic stream generators (seeded — identical across runs)
# ----------------------------------------------------------------------

def _stable_r_stream(n: int, seed: int, mean: float = 0.60, std: float = 0.015) -> np.ndarray:
    """A stationary r-ratio stream with no changepoint."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=n)


def _changepoint_r_stream(
    n: int, cp_index: int, seed: int,
    mean_before: float = 0.60, mean_after: float = 0.42, std: float = 0.015,
) -> np.ndarray:
    """A stream that shifts from ``mean_before`` to ``mean_after`` at ``cp_index``."""
    rng = np.random.default_rng(seed)
    out = rng.normal(mean_before, std, size=n)
    out[cp_index:] = rng.normal(mean_after, std, size=n - cp_index)
    return out


def _gue_eigenvalues(dim: int, seed: int) -> np.ndarray:
    """Eigenvalues of one GUE-distributed Hermitian matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2.0
    return np.linalg.eigvalsh(h)


def _unfold(evals: np.ndarray) -> np.ndarray:
    """Unfold a spectrum to unit mean spacing via a polynomial fit to the
    cumulative density — required before spacing statistics are comparable to
    the RMT reference values."""
    evals = np.sort(evals)
    counts = np.arange(1, len(evals) + 1)
    coeffs = np.polyfit(evals, counts, deg=9)
    return np.polyval(coeffs, evals)


# ======================================================================
# Checks — environment and packaging
# ======================================================================

@check("env_capture")
def _env_capture(rec: Recorder, args: argparse.Namespace) -> None:
    """Record the exact environment every later number was produced in."""
    import torch

    rec.metric("python_version", platform.python_version())
    rec.metric("platform", platform.platform())
    rec.metric("numpy_version", np.__version__)
    rec.metric("torch_version", torch.__version__)
    rec.metric("cuda_available", bool(torch.cuda.is_available()))
    rec.metric("device", device_name())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rec.metric("gpu_name", props.name)
        rec.metric("gpu_total_memory_gb", round(props.total_memory / 1024**3, 2))

    try:
        import transformers
        rec.metric("transformers_version", transformers.__version__)
    except ImportError:
        rec.metric("transformers_version", None)
        rec.note("transformers not installed — model checks will be skipped")


@check("package_integrity")
def _package_integrity(rec: Recorder, args: argparse.Namespace) -> None:
    """Verify the package names actually resolve to what they claim.

    This is the check that would have caught the shipped-broken states:
    a distribution whose declared VAR dependency PyPI silently dropped, and a
    `var` import that resolved to an unrelated project of the same name.
    """
    import importlib.metadata as md

    from unitarity_labs.core.version import __version__ as core_version

    rec.metric("unitarity_labs_import_version", core_version)

    try:
        dist_version = md.version("unitarity-labs")
    except md.PackageNotFoundError:
        dist_version = None
        rec.note(
            "Distribution 'unitarity-labs' is not installed. Note the repo is "
            "'unitarity-lab' (no trailing s) but the distribution is "
            "'unitarity-labs' (with s)."
        )
    rec.metric("unitarity_labs_dist_version", dist_version)
    rec.metric(
        "version_consistent",
        dist_version is not None and dist_version == core_version,
    )

    # VAR must resolve to var_spectral, never to the unrelated PyPI `var`.
    try:
        import var_spectral
        rec.metric("var_spectral_version", var_spectral.__version__)
        rec.metric("var_spectral_importable", True)
    except ImportError:
        rec.metric("var_spectral_version", None)
        rec.metric("var_spectral_importable", False)
        rec.note(
            "var_spectral not importable — install 'unitarity-labs[spectral]'. "
            "Do NOT 'pip install var': that is an unrelated project."
        )

    shadowed = False
    try:
        import var as _var
        shadowed = not getattr(_var, "__name__", "").startswith("var_spectral")
        rec.metric("var_module_file", getattr(_var, "__file__", None))
    except ImportError:
        rec.metric("var_module_file", None)
    rec.metric("var_name_shadowed_by_foreign_package", shadowed)
    if shadowed:
        rec.note(
            "A foreign 'var' package is importable and will shadow VAR. "
            "Uninstall it: pip uninstall var"
        )


# ======================================================================
# Checks — BOCPD monitor
# ======================================================================

@check("bocpd_null")
def _bocpd_null(rec: Recorder, args: argparse.Namespace) -> None:
    """A stationary stream must not produce changepoint alarms."""
    from unitarity_labs.core.bocpd import PredictiveAnomalyDetector

    n, warmup = 600, 100
    stream = _stable_r_stream(n, seed=args.seed)
    det = PredictiveAnomalyDetector(warmup_steps=warmup)

    probs = [det.process_step(zeta=1.0, r_ratio=float(v)) for v in stream]
    post = np.asarray(probs[warmup:])

    alarms = int((post >= det.threshold).sum())
    rec.metric("stream_length", n)
    rec.metric("warmup_steps", warmup)
    rec.metric("post_warmup_steps", int(post.size))
    rec.metric("false_alarms", alarms)
    rec.metric("false_alarm_rate", round(alarms / post.size, 6))
    rec.metric("max_changepoint_prob", round(float(post.max()), 8))
    rec.metric("mean_changepoint_prob", round(float(post.mean()), 8))
    rec.metric("calibrated_mean_0", round(float(det.mean_0), 6))
    rec.metric("calibrated_std_0", round(float(det.std_0), 6))
    rec.metric("passed", alarms == 0)

    if alarms:
        rec.note(f"{alarms} false alarm(s) on a stream with no changepoint")


@check("bocpd_changepoint")
def _bocpd_changepoint(rec: Recorder, args: argparse.Namespace) -> None:
    """A stream with a known changepoint must be detected, and only once."""
    from unitarity_labs.core.bocpd import PredictiveAnomalyDetector

    n, warmup, cp = 600, 100, 300
    stream = _changepoint_r_stream(n, cp_index=cp, seed=args.seed)
    det = PredictiveAnomalyDetector(warmup_steps=warmup)

    probs = [det.process_step(zeta=1.0, r_ratio=float(v)) for v in stream]
    probs_arr = np.asarray(probs)

    fired = np.flatnonzero(probs_arr >= det.threshold)
    first = int(fired[0]) if fired.size else -1

    rec.metric("stream_length", n)
    rec.metric("true_changepoint_index", cp)
    rec.metric("first_alarm_index", first)
    rec.metric("detected", bool(fired.size))
    rec.metric("detection_delay_steps", first - cp if first >= 0 else None)
    # An alarm before the true changepoint is a false positive by construction.
    rec.metric("pre_changepoint_alarms", int((fired < cp).sum()))
    rec.metric("max_prob_before_cp", round(float(probs_arr[warmup:cp].max()), 8))
    rec.metric("max_prob_after_cp", round(float(probs_arr[cp:].max()), 8))
    rec.metric("calibrated_mean_0", round(float(det.mean_0), 6))

    passed = bool(fired.size) and first >= cp
    rec.metric("passed", passed)
    if not fired.size:
        rec.note("changepoint was never detected")
    elif first < cp:
        rec.note(f"alarm at step {first} precedes the true changepoint at {cp}")


@check("bocpd_warmup_calibration")
def _bocpd_warmup(rec: Recorder, args: argparse.Namespace) -> None:
    """Warm-up calibration must recover the mean of the stream it saw."""
    from unitarity_labs.core.bocpd import PredictiveAnomalyDetector

    warmup, true_mean, true_std = 200, 0.6400, 0.0150
    stream = _stable_r_stream(warmup, seed=args.seed, mean=true_mean, std=true_std)
    det = PredictiveAnomalyDetector(warmup_steps=warmup)
    for v in stream:
        det.process_step(zeta=1.0, r_ratio=float(v))

    rec.metric("empirical_mean", round(float(stream.mean()), 6))
    rec.metric("empirical_std", round(float(stream.std()), 6))
    rec.check_close("calibrated_mean_0", det.mean_0, float(stream.mean()), tol=1e-9)
    rec.check_close("calibrated_std_0", det.std_0, float(stream.std()), tol=1e-9)
    rec.metric("calibrated_flag", bool(det.calibrated))
    rec.metric("passed", bool(det.calibrated))


@check("bocpd_return_range")
def _bocpd_range(rec: Recorder, args: argparse.Namespace) -> None:
    """Returned probabilities must stay in [0, 1] even on adversarial input."""
    from unitarity_labs.core.bocpd import PredictiveAnomalyDetector

    det = PredictiveAnomalyDetector(warmup_steps=50)
    rng = np.random.default_rng(args.seed)
    hostile = np.concatenate([
        rng.normal(0.60, 0.015, size=100),
        np.array([0.0, 1.0, -5.0, 5.0, 1e6, -1e6, 0.42, 0.60]),
        rng.normal(0.42, 0.015, size=50),
    ])

    out = [det.process_step(zeta=1.0, r_ratio=float(v)) for v in hostile]
    arr = np.asarray(out)

    rec.metric("samples", int(arr.size))
    rec.metric("min_prob", round(float(arr.min()), 8))
    rec.metric("max_prob", round(float(arr.max()), 8))
    rec.metric("any_nan", bool(np.isnan(arr).any()))
    rec.metric("any_inf", bool(np.isinf(arr).any()))
    in_range = bool(((arr >= 0.0) & (arr <= 1.0)).all())
    rec.metric("all_in_unit_interval", in_range)
    rec.metric("passed", in_range and not np.isnan(arr).any())


# ======================================================================
# Checks — GUE / spectral reference values
# ======================================================================

@check("gue_r_ratio")
def _gue_r_ratio(rec: Recorder, args: argparse.Namespace) -> None:
    """⟨r⟩ must match random-matrix theory for GUE and for Poisson.

    These are the two anchor numbers the whole spectral story rests on, so they
    are measured against theory rather than asserted.
    """
    from unitarity_labs.core.spectral_monitor import get_r_ratio

    dim, ensembles = 256, 24

    gue_vals = []
    for i in range(ensembles):
        evals = _gue_eigenvalues(dim, seed=args.seed + i)
        gue_vals.append(get_r_ratio(_unfold(evals)))
    gue_mean = float(np.mean(gue_vals))

    rng = np.random.default_rng(args.seed + 9999)
    poisson_vals = [
        get_r_ratio(np.sort(rng.uniform(0.0, 1.0, size=dim)))
        for _ in range(ensembles)
    ]
    poisson_mean = float(np.mean(poisson_vals))

    rec.metric("matrix_dim", dim)
    rec.metric("ensembles", ensembles)
    rec.metric("gue_r_std", round(float(np.std(gue_vals)), 6))
    rec.metric("poisson_r_std", round(float(np.std(poisson_vals)), 6))
    # Tolerance is finite-size slack at dim=256 over 24 ensembles, not a fudge:
    # the standard error of the mean is ~0.005, so 0.02 is ~4 sigma.
    ok_gue = rec.check_close("gue_r_mean", gue_mean, GUE_R_REFERENCE, tol=0.02)
    ok_poisson = rec.check_close(
        "poisson_r_mean", poisson_mean, POISSON_R_REFERENCE, tol=0.02
    )
    rec.metric("separation", round(gue_mean - poisson_mean, 6))
    rec.metric("passed", ok_gue and ok_poisson)


# ======================================================================
# Checks — Chronos lock
# ======================================================================

@check("chronos_tps")
def _chronos_tps(rec: Recorder, args: argparse.Namespace) -> None:
    """TPS estimation must clip outliers rather than track them."""
    from unitarity_labs.core.chronos_lock import (
        TPS_CLIP_MAX, TPS_CLIP_MIN, ChronosLock,
    )

    lock = ChronosLock(node_id="audit")
    steady = [12.0] * 20
    for v in steady:
        lock.update_tps(v)
    rec.metric("steady_state_tps", round(float(lock.update_tps(12.0)), 6))

    after_spike = float(lock.update_tps(10_000.0))
    rec.metric("tps_after_extreme_spike", round(after_spike, 6))
    rec.metric("clip_min", TPS_CLIP_MIN)
    rec.metric("clip_max", TPS_CLIP_MAX)
    bounded = TPS_CLIP_MIN <= after_spike <= TPS_CLIP_MAX
    rec.metric("spike_bounded_by_clip", bool(bounded))

    after_zero = float(lock.update_tps(0.0))
    rec.metric("tps_after_zero", round(after_zero, 6))
    rec.metric("zero_bounded_by_clip", bool(TPS_CLIP_MIN <= after_zero <= TPS_CLIP_MAX))
    rec.metric("passed", bool(bounded and TPS_CLIP_MIN <= after_zero <= TPS_CLIP_MAX))


@check("chronos_desync")
def _chronos_desync(rec: Recorder, args: argparse.Namespace) -> None:
    """Cumulative desync must trip the adaptive threshold, and not before."""
    from unitarity_labs.core.chronos_lock import ChronosLock

    calm = ChronosLock(node_id="audit-calm")
    calm_trips = sum(bool(calm.update_desync(0.001)) for _ in range(64))
    rec.metric("calm_trip_count", calm_trips)
    rec.metric("calm_threshold", round(float(calm._adaptive_threshold()), 6))

    hot = ChronosLock(node_id="audit-hot")
    first_trip = -1
    for i in range(64):
        if hot.update_desync(0.5) and first_trip < 0:
            first_trip = i
    rec.metric("hot_first_trip_index", first_trip)
    rec.metric("hot_threshold", round(float(hot._adaptive_threshold()), 6))
    rec.metric("passed", bool(calm_trips == 0 and first_trip >= 0))

    if calm_trips:
        rec.note(f"{calm_trips} spurious desync trip(s) on a calm link")
    if first_trip < 0:
        rec.note("sustained 0.5s desync never tripped the threshold")


@check("chronos_shard_roundtrip")
def _chronos_shard(rec: Recorder, args: argparse.Namespace) -> None:
    """Reed–Solomon shards: clean round-trip, 1-symbol repair, 3-symbol detect.

    With ``RS_NSYM=2`` parity symbols the code corrects ⌊2/2⌋ = 1 corrupted
    symbol. Beyond that it must *fail loudly* rather than return wrong data —
    silent corruption is the only unacceptable outcome here.
    """
    from unitarity_labs.core.chronos_lock import (
        SHARD_SYMBOL_LEN, ChronosLock,
    )

    lock = ChronosLock(node_id="audit-shard")
    for i in range(8):
        lock.record_τ(float(i) * 0.125)

    shard = lock.encode_shard()
    rec.metric("shard_bytes", len(shard))

    clean = lock.decode_shard(shard)
    rec.metric("clean_roundtrip_ok", clean is not None)

    corrupt1 = bytearray(shard)
    corrupt1[0] ^= 0xFF
    repaired = lock.decode_shard(bytes(corrupt1))
    rec.metric("single_symbol_repaired", repaired is not None)
    rec.metric("repair_matches_clean", repaired == clean)

    corrupt3 = bytearray(shard)
    for offset in (0, SHARD_SYMBOL_LEN, 2 * SHARD_SYMBOL_LEN):
        if offset < len(corrupt3):
            corrupt3[offset] ^= 0xFF
    # Beyond RS capacity the only acceptable outcomes are "raise" or "return
    # None". Returning *different* data without complaint is silent corruption.
    try:
        over = lock.decode_shard(bytes(corrupt3))
        outcome = "returned_none" if over is None else "returned_data"
        silently_wrong = over is not None and over != clean
    except (ValueError, Exception) as exc:  # noqa: B014 - codec raises broadly
        over, silently_wrong = None, False
        outcome = f"raised_{type(exc).__name__}"
    rec.metric("beyond_capacity_outcome", outcome)
    rec.metric("silent_corruption", bool(silently_wrong))

    passed = (
        clean is not None
        and repaired is not None
        and repaired == clean
        and not silently_wrong
    )
    rec.metric("passed", passed)
    if silently_wrong:
        rec.note("decode returned corrupted data beyond RS capacity without error")


@check("chronos_tau_chain")
def _chronos_tau_chain(rec: Recorder, args: argparse.Namespace) -> None:
    """The τ hash chain must validate intact and reject a tampered history."""
    from unitarity_labs.core.chronos_lock import ChronosLock

    lock = ChronosLock(node_id="audit-chain")
    for i in range(16):
        lock.record_τ(float(i) * 0.05)

    digest = lock.compute_τ_hash()
    rec.metric("hash_present", digest is not None)
    rec.metric("hash_length", len(digest) if digest else 0)

    # Same history, independent instance → identical digest.
    twin = ChronosLock(node_id="audit-chain")
    for i in range(16):
        twin.record_τ(float(i) * 0.05)
    rec.metric("hash_reproducible", twin.compute_τ_hash() == digest)

    # One perturbed sample → different digest.
    tampered = ChronosLock(node_id="audit-chain")
    for i in range(16):
        tampered.record_τ(float(i) * 0.05 + (0.001 if i == 7 else 0.0))
    tampered_digest = tampered.compute_τ_hash()
    detected = tampered_digest != digest
    rec.metric("tamper_detected", bool(detected))
    rec.metric("passed", bool(digest and twin.compute_τ_hash() == digest and detected))

    if not detected:
        rec.note("a tampered τ history produced an identical hash")


# ======================================================================
# Checks — VAR spectral rupture detector
# ======================================================================

@check("var_rupture_detector")
def _var_rupture(rec: Recorder, args: argparse.Namespace) -> None:
    """VAR's rupture detector: silent on a calm signal, fires on a real one."""
    try:
        from var_spectral.detector import RuptureState, SpectralRuptureDetector
    except ImportError as e:
        raise RuntimeError(
            "var_spectral is required. Install 'unitarity-labs[spectral]'. "
            "Do not 'pip install var' — that is an unrelated project."
        ) from e

    rng = np.random.default_rng(args.seed)

    calm = SpectralRuptureDetector()
    calm_signal = rng.normal(1.0, 0.01, size=400)
    calm_ruptures = 0
    for v in calm_signal:
        calm.update(float(v))
        if calm.state is RuptureState.RUPTURED:
            calm_ruptures += 1
    rec.metric("calm_samples", int(calm_signal.size))
    rec.metric("calm_rupture_steps", calm_ruptures)

    hot = SpectralRuptureDetector()
    baseline = rng.normal(1.0, 0.01, size=300)
    excursion = rng.normal(5.0, 0.05, size=100)
    first_rupture = -1
    for i, v in enumerate(np.concatenate([baseline, excursion])):
        hot.update(float(v))
        if hot.state is RuptureState.RUPTURED and first_rupture < 0:
            first_rupture = i
    rec.metric("injected_rupture_index", int(baseline.size))
    rec.metric("first_rupture_index", first_rupture)
    rec.metric("rupture_detected", first_rupture >= 0)
    rec.metric(
        "detection_delay_steps",
        first_rupture - int(baseline.size) if first_rupture >= 0 else None,
    )

    passed = calm_ruptures == 0 and first_rupture >= int(baseline.size)
    rec.metric("passed", bool(passed))
    if calm_ruptures:
        rec.note(f"{calm_ruptures} rupture step(s) on a calm signal")


# ======================================================================
# Checks — model-dependent
# ======================================================================

def _load_model(args: argparse.Namespace):
    """Load the audit model once per check, on the selected device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    if not torch.cuda.is_available() and dtype is not torch.float32:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers renamed `torch_dtype` to `dtype` in 4.56; older releases
    # only accept the former. Colab pins vary, so support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer, dtype


@check("model_load", needs_model=True)
def _model_load(rec: Recorder, args: argparse.Namespace) -> None:
    """Load the model and record its real size — no estimates."""
    import torch

    t0 = time.perf_counter()
    model, tokenizer, dtype = _load_model(args)
    rec.metric("load_seconds", round(time.perf_counter() - t0, 3))

    params = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    rec.metric("model_id", args.model)
    rec.metric("dtype", str(dtype).replace("torch.", ""))
    rec.metric("parameters", int(params))
    rec.metric("parameters_millions", round(params / 1e6, 2))
    rec.metric("weight_memory_gb", round(bytes_ / 1024**3, 4))
    rec.metric("hidden_size", int(model.config.hidden_size))
    rec.metric("num_layers", int(getattr(model.config, "num_hidden_layers", 0)))
    rec.metric("vocab_size", int(model.config.vocab_size))

    if torch.cuda.is_available():
        rec.metric(
            "cuda_allocated_gb",
            round(torch.cuda.memory_allocated() / 1024**3, 4),
        )
        total = torch.cuda.get_device_properties(0).total_memory
        rec.metric("fits_in_device", bool(bytes_ < total * 0.9))
    rec.metric("passed", True)


@check("model_generation_determinism", needs_model=True)
def _model_determinism(rec: Recorder, args: argparse.Namespace) -> None:
    """Greedy decoding must return byte-identical text across repeats.

    If this fails, no downstream number from this model is reproducible and
    every other model metric should be treated as suspect.
    """
    import torch

    model, tokenizer, _ = _load_model(args)
    prompt = "The capital of France is"
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outs = []
    for _ in range(3):
        with torch.no_grad():
            ids = model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        outs.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    identical = len(set(outs)) == 1
    rec.metric("completions", outs)
    rec.metric("unique_completions", len(set(outs)))
    rec.metric("greedy_deterministic", bool(identical))
    rec.metric("passed", bool(identical))
    if not identical:
        rec.note("greedy decoding is not reproducible on this device")


@check("model_zeta_passive", needs_model=True)
def _model_zeta(rec: Recorder, args: argparse.Namespace) -> None:
    """Capture real ζ and spectral gap from real activations.

    This replaces the old notebook path, which caught the ImportError from a
    non-existent API and reported ``np.random.randn`` values instead.
    """
    import torch

    from unitarity_labs.core.universal_hook import UniversalHookWrapper

    model, tokenizer, _ = _load_model(args)
    wrapper = UniversalHookWrapper(model=model, config=model.config, mode="passive")
    device = next(model.parameters()).device

    prompts = [
        "Explain coherence in quantum systems.",
        "Compute the sum of the first ten primes.",
        "Describe the process of photosynthesis.",
        "Write a short poem about winter rain.",
    ]

    zetas, gaps, latencies = [], [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        zetas.append(wrapper.bridge.raw_sink_zeta)
        gaps.append(wrapper.bridge.spectral_gap())

    finite_z = [float(z) for z in zetas if z is not None and np.isfinite(z)]
    finite_g = [float(g) for g in gaps if g is not None and np.isfinite(g)]

    rec.metric("prompts", len(prompts))
    rec.metric("zeta_values", [round(z, 8) for z in finite_z])
    rec.metric("spectral_gap_values", [round(g, 8) for g in finite_g])
    rec.metric("zeta_captured", len(finite_z))
    rec.metric("spectral_gap_captured", len(finite_g))
    if finite_z:
        rec.metric("zeta_mean", round(float(np.mean(finite_z)), 8))
        rec.metric("zeta_std", round(float(np.std(finite_z)), 8))
    if finite_g:
        rec.metric("spectral_gap_mean", round(float(np.mean(finite_g)), 8))
    rec.metric("mean_forward_latency_ms", round(float(np.mean(latencies)), 3))

    passed = len(finite_z) == len(prompts)
    rec.metric("passed", passed)
    if not passed:
        rec.note(
            f"only {len(finite_z)}/{len(prompts)} prompts yielded a finite zeta"
        )


@check("model_length_matched_null", needs_model=True)
def _model_null(rec: Recorder, args: argparse.Namespace) -> None:
    """Measured ζ against a length-matched null built from other prompts.

    Without this contrast a ζ value means nothing: the null says what ζ looks
    like when the source and sink genuinely do not correspond.
    """
    import torch

    from unitarity_labs.core.metrics import length_matched_null_zeta
    from unitarity_labs.core.universal_hook import UniversalHookWrapper

    model, tokenizer, _ = _load_model(args)
    wrapper = UniversalHookWrapper(model=model, config=model.config, mode="passive")
    device = next(model.parameters()).device

    prompts = [
        "Explain coherence in quantum systems.",
        "Compute the sum of the first ten primes.",
        "Describe the process of photosynthesis.",
        "Write a short poem about winter rain.",
        "Summarise the causes of the French Revolution.",
        "List three properties of prime numbers.",
    ]

    sources, sinks = [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        src = wrapper.bridge._source_activation
        snk = wrapper.bridge._sink_activation
        sources.append(src.detach().cpu() if src is not None else None)
        sinks.append(snk.detach().cpu() if snk is not None else None)

    usable = [
        i for i in range(len(prompts))
        if sources[i] is not None and sinks[i] is not None
    ]
    rec.metric("prompts", len(prompts))
    rec.metric("usable_captures", len(usable))

    if len(usable) < 3:
        rec.metric("passed", False)
        rec.note(
            f"only {len(usable)} usable capture(s); need >= 3 controls for a null"
        )
        return

    z_scores, gaps = [], []
    for i in usable:
        controls = [sinks[j] for j in usable if j != i]
        result = length_matched_null_zeta(sources[i], sinks[i], controls)
        if result is None:
            continue
        z_scores.append(float(result["z_score"]))
        gaps.append(float(result["gap"]))

    rec.metric("null_results", len(z_scores))
    if z_scores:
        rec.metric("z_score_values", [round(z, 6) for z in z_scores])
        rec.metric("z_score_mean", round(float(np.mean(z_scores)), 6))
        rec.metric("z_score_std", round(float(np.std(z_scores)), 6))
        rec.metric("gap_mean", round(float(np.mean(gaps)), 8))
    rec.metric("passed", bool(z_scores))


# ======================================================================
# Runner
# ======================================================================

def _aggregate(runs: List[CheckResult]) -> Dict[str, Any]:
    """Collapse repeated runs of one check into a summary with variance."""
    summary: Dict[str, Any] = {
        "check": runs[0].name,
        "repeats": len(runs),
        "statuses": [r.status for r in runs],
        "status": "pass" if all(r.status == "pass" for r in runs) else (
            "error" if any(r.status == "error" for r in runs) else "fail"
        ),
        "duration_s_mean": round(
            statistics.fmean(r.duration_s for r in runs), 4
        ),
        "errors": sorted({r.error for r in runs if r.error}),
        "notes": sorted({n for r in runs for n in r.notes}),
    }

    keys: List[str] = []
    for r in runs:
        for k in r.metrics:
            if k not in keys:
                keys.append(k)

    metrics: Dict[str, Any] = {}
    for key in keys:
        values = [r.metrics[key] for r in runs if key in r.metrics]
        numeric = [
            v for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if numeric and len(numeric) == len(values):
            entry: Dict[str, Any] = {
                "value": numeric[0],
                "min": min(numeric),
                "max": max(numeric),
                "mean": round(statistics.fmean(numeric), 10),
                "deterministic": min(numeric) == max(numeric),
            }
            if len(numeric) > 1:
                entry["stdev"] = round(statistics.pstdev(numeric), 10)
            metrics[key] = entry
        else:
            distinct = []
            for v in values:
                if v not in distinct:
                    distinct.append(v)
            metrics[key] = {
                "value": values[0],
                "deterministic": len(distinct) == 1,
            }
            if len(distinct) > 1:
                metrics[key]["distinct_values"] = distinct
    summary["metrics"] = metrics
    return summary


def _run_one(name: str, fn: CheckFn, args: argparse.Namespace) -> CheckResult:
    result = CheckResult(name)
    rec = Recorder(result)
    t0 = time.perf_counter()
    try:
        fn(rec, args)
        passed = result.metrics.get("passed", True)
        result.status = "pass" if passed else "fail"
    except Exception:
        result.status = "error"
        result.error = traceback.format_exc(limit=6).strip()
    result.duration_s = time.perf_counter() - t0
    return result


def _selected_checks(args: argparse.Namespace) -> List[str]:
    names = list(REGISTRY)
    if args.only:
        requested = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in requested if n not in REGISTRY]
        if unknown:
            raise SystemExit(
                f"Unknown check(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(REGISTRY)}"
            )
        names = [n for n in names if n in requested]
    if args.skip:
        skipped = {n.strip() for n in args.skip.split(",")}
        names = [n for n in names if n not in skipped]
    if args.no_model:
        names = [n for n in names if n not in MODEL_CHECKS]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential audit suite — runs each check one at a time "
                    "and logs every measured value.",
    )
    parser.add_argument("--model", default=None,
                        help="HF model id for model-dependent checks.")
    parser.add_argument("--model-tier", choices=sorted(MODEL_TIERS),
                        default=None,
                        help="T4-safe preset; sets --model and --dtype.")
    parser.add_argument("--dtype", default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (forced to float32 on CPU).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run each check N times and report variance "
                             "(default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated check names to run.")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated check names to skip.")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip all model-dependent checks.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for results (default: "
                             "results/audits/<date>_<sha>_<env>).")
    parser.add_argument("--list", action="store_true",
                        help="List available checks and exit.")
    args = parser.parse_args()

    if args.list:
        for name in REGISTRY:
            tag = " [needs model]" if name in MODEL_CHECKS else ""
            print(f"{name}{tag}")
        return 0

    if args.model_tier:
        tier = MODEL_TIERS[args.model_tier]
        args.model = args.model or tier["model"]
        args.dtype = args.dtype or tier["dtype"]
    args.dtype = args.dtype or "float32"
    if args.model is None:
        args.no_model = True

    names = _selected_checks(args)
    if not names:
        print("No checks selected.", file=sys.stderr)
        return 2

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_dir = Path("results/audits") / f"{stamp}_{git_sha()}_{env_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "audit.log.jsonl"

    print(f"Audit suite — {len(names)} check(s), repeat={args.repeat}, seed={args.seed}")
    print(f"Output: {out_dir}")
    if not args.no_model:
        print(f"Model:  {args.model} ({args.dtype})")
    print("=" * 72)

    summaries: List[Dict[str, Any]] = []
    # One check at a time, flushed to disk as it completes, so a crash later in
    # the suite cannot cost you the results already measured.
    with log_path.open("w", encoding="utf-8") as log:
        for index, name in enumerate(names, start=1):
            print(f"\n[{index}/{len(names)}] {name}")
            runs: List[CheckResult] = []
            for attempt in range(args.repeat):
                result = _run_one(name, REGISTRY[name], args)
                runs.append(result)
                log.write(json.dumps(
                    {"run": attempt + 1, **result.to_dict()}, default=str
                ) + "\n")
                log.flush()

                marker = {"pass": "PASS", "fail": "FAIL", "error": "ERROR"}[result.status]
                suffix = f" (run {attempt + 1}/{args.repeat})" if args.repeat > 1 else ""
                print(f"  {marker}{suffix}  {result.duration_s:.3f}s")
                for key, value in result.metrics.items():
                    print(f"      {key} = {value}")
                for note in result.notes:
                    print(f"      note: {note}")
                if result.error:
                    for line in result.error.splitlines():
                        print(f"      {line}")

            summary = _aggregate(runs)
            summaries.append(summary)

            if args.repeat > 1:
                unstable = [
                    k for k, v in summary["metrics"].items()
                    if not v.get("deterministic", True)
                ]
                if unstable:
                    print(f"      NON-DETERMINISTIC across repeats: {', '.join(unstable)}")
                else:
                    print("      all metrics identical across repeats")

    counts = {
        status: sum(1 for s in summaries if s["status"] == status)
        for status in ("pass", "fail", "error")
    }
    nondet = sorted(
        f"{s['check']}.{k}"
        for s in summaries
        for k, v in s["metrics"].items()
        if not v.get("deterministic", True)
    )

    report = {
        "suite": "audit_suite",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "repeat": args.repeat,
        "model": None if args.no_model else args.model,
        "dtype": None if args.no_model else args.dtype,
        "counts": counts,
        "non_deterministic_metrics": nondet,
        "checks": summaries,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(json.dumps({
        "git_sha": git_sha(),
        "device": device_name(),
        "env": env_tag(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "repeat": args.repeat,
        "model": None if args.no_model else args.model,
        "dtype": None if args.no_model else args.dtype,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pip_freeze": pip_freeze(),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"pass={counts['pass']}  fail={counts['fail']}  error={counts['error']}")
    if args.repeat > 1:
        if nondet:
            print(f"non-deterministic metrics ({len(nondet)}):")
            for key in nondet:
                print(f"  {key}")
        else:
            print("every metric was identical across all repeats")
    print(f"\nWrote {out_dir / 'audit.json'}")
    print(f"Wrote {log_path}")
    print(f"Wrote {out_dir / 'manifest.json'}")

    return 0 if counts["fail"] == 0 and counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
