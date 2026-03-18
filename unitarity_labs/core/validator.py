"""
core/validator.py — Benchmark comparison and text-based metric extraction.
=========================================================================
Defines the reference spectral benchmark and provides health-evaluation
logic that computes the *Spectral Divergence* of observed statistics from
that benchmark.

The stabilising constraint follows the Berry–Keating / Kar Hamiltonian at
k = 1, which fixes the expected level-repulsion statistics of a healthy
transport manifold (Kar, "Random matrix theory and the Riemann zeros",
k = 1 sector).  Under k = 1 the pair-correlation function converges to
the GUE form, giving ⟨r⟩ ≈ 0.60 for an ideal system and ≈ 0.58 for a
practical large-scale model.

Provides
--------
* ``GROK_4_MARCH_2026_BENCHMARK`` — reference statistics.
* ``evaluate_model_health``  — spectral divergence from benchmark.
* ``parse_metrics_from_text`` — regex extraction of ζ and ⟨r⟩ from
  free-form text (CLI / audit-table ingestion).
* ``log_audit`` — append a JSON audit record to
  ``benchmarks/audit_logs/``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Benchmark (k = 1 Kar–Berry–Keating sector) ──────────────────────
GROK_4_MARCH_2026_BENCHMARK: dict = {
    "r_ratio": 0.58,
    "zeta": 0.85,
    "frobenius_stability": 0.21,
}
"""Reference spectral profile.

* ⟨r⟩ = 0.58 — expected GUE-class gap ratio under the k = 1
  Berry–Keating / Kar Hamiltonian constraint.
* ζ = 0.85 — manifold coherence (cosine similarity).
* Frobenius stability = 0.21 — ‖A − I‖_F for the transport matrix.
"""

# Default audit-log directory (relative to repo root).
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "audit_logs"


# ── Data containers ──────────────────────────────────────────────────
@dataclass
class ModelStats:
    """Observed spectral statistics for a model or session."""

    r_ratio: Optional[float] = None
    zeta: Optional[float] = None
    frobenius_stability: Optional[float] = None


@dataclass
class HealthReport:
    """Result of comparing observed stats against the benchmark."""

    delta_r: Optional[float] = None
    delta_zeta: Optional[float] = None
    delta_frobenius: Optional[float] = None
    spectral_divergence: float = 0.0
    passed: bool = True
    details: str = ""


# ── Core evaluation ──────────────────────────────────────────────────
def evaluate_model_health(
    input_stats: ModelStats,
    benchmark: Optional[dict] = None,
) -> HealthReport:
    """Compute Spectral Divergence from the reference benchmark.

    The divergence is the Euclidean distance in the (⟨r⟩, ζ, ‖·‖_F)
    space between the observed statistics and the benchmark, weighted so
    that each axis contributes proportionally to its reference value:

        D = sqrt( (Δr/r_ref)² + (Δζ/ζ_ref)² + (Δf/f_ref)² )

    where Δx = x_obs − x_ref.

    Under the k = 1 Kar constraint the gap ratio is the dominant health
    signal; divergence > 1.0 indicates the system has left the stable
    spectral basin.

    Parameters
    ----------
    input_stats : ModelStats
        Observed statistics (any field may be ``None`` → skipped).
    benchmark : dict, optional
        Override the default benchmark dictionary.

    Returns
    -------
    HealthReport
    """
    ref = benchmark or GROK_4_MARCH_2026_BENCHMARK
    terms = []
    report = HealthReport()

    if input_stats.r_ratio is not None:
        report.delta_r = input_stats.r_ratio - ref["r_ratio"]
        terms.append((report.delta_r / ref["r_ratio"]) ** 2)

    if input_stats.zeta is not None:
        report.delta_zeta = input_stats.zeta - ref["zeta"]
        terms.append((report.delta_zeta / ref["zeta"]) ** 2)

    if input_stats.frobenius_stability is not None:
        report.delta_frobenius = input_stats.frobenius_stability - ref["frobenius_stability"]
        terms.append((report.delta_frobenius / ref["frobenius_stability"]) ** 2)

    if terms:
        report.spectral_divergence = sum(terms) ** 0.5
    report.passed = report.spectral_divergence <= 1.0
    report.details = (
        f"Divergence={report.spectral_divergence:.4f} "
        f"(Δr={report.delta_r}, Δζ={report.delta_zeta}, "
        f"Δf={report.delta_frobenius})"
    )
    return report


# ── Text parsing ─────────────────────────────────────────────────────
# Patterns intentionally liberal to handle Unicode symbols, markdown,
# LaTeX fragments, and plain-ASCII representations.

_ZETA_RE = re.compile(
    r"(?:ζ|zeta|\\zeta|coherence)"
    r"\s*[=:≈~]\s*"
    r"([+-]?\d+\.?\d*)",
    re.IGNORECASE,
)

_R_RATIO_RE = re.compile(
    r"(?:⟨r⟩|<r>|r[_-]?ratio|gap[_\s-]*ratio|\\langle\s*r\s*\\rangle)"
    r"\s*[=:≈~]\s*"
    r"([+-]?\d+\.?\d*)",
    re.IGNORECASE,
)

_FROB_RE = re.compile(
    r"(?:frobenius|frob|\\|A\s*-\s*I\\|_F)"
    r"[_\s-]*(?:stability|dist(?:ance)?)?"
    r"\s*[=:≈~]\s*"
    r"([+-]?\d+\.?\d*)",
    re.IGNORECASE,
)


def parse_metrics_from_text(text: str) -> ModelStats:
    """Extract ζ, ⟨r⟩, and Frobenius values from free-form text.

    Handles markdown tables, LaTeX fragments, plain ASCII, and Unicode.

    Parameters
    ----------
    text : str
        Raw text from an external model's output, audit table, etc.

    Returns
    -------
    ModelStats
        Populated with whatever metrics were found (others are ``None``).
    """
    stats = ModelStats()

    m = _ZETA_RE.search(text)
    if m:
        stats.zeta = float(m.group(1))

    m = _R_RATIO_RE.search(text)
    if m:
        stats.r_ratio = float(m.group(1))

    m = _FROB_RE.search(text)
    if m:
        stats.frobenius_stability = float(m.group(1))

    return stats


# ── JSON audit logging ───────────────────────────────────────────────
def log_audit(
    report: HealthReport,
    stats: Optional[ModelStats] = None,
    *,
    log_dir: Optional[Path] = None,
    tag: str = "",
) -> Path:
    """Append an audit record as a timestamped JSON file.

    Parameters
    ----------
    report : HealthReport
    stats : ModelStats, optional
        The input stats that produced the report.
    log_dir : Path, optional
        Override the default ``benchmarks/audit_logs/`` directory.
    tag : str
        Optional short label included in the filename.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    out_dir = log_dir or _DEFAULT_LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"_{tag}" if tag else ""
    filename = f"audit_{ts}{slug}.json"
    path = out_dir / filename

    record = {
        "timestamp": ts,
        "tag": tag,
        "benchmark": GROK_4_MARCH_2026_BENCHMARK,
        "observed": asdict(stats) if stats else None,
        "report": asdict(report),
    }

    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    logger.info("Audit log written to %s", path)
    return path
