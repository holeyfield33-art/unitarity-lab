"""Runner — execute a suite against a provider, score it, record it.

Mirrors the rest of the repo's reproducibility discipline: every result set is
written next to a manifest capturing provider, model, suite path + content hash,
UTC timestamp, and versions, so any number can be traced to exactly what
produced it. Nothing is precomputed — results come only from a real run.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .providers import Provider
from .scoring import CaseResult, score_case
from .suite import TestCase, suite_sha256


@dataclass
class RunSummary:
    provider: str
    model: str
    total: int
    passed: int
    failed: int
    errored: int
    by_category: dict = field(default_factory=dict)
    results: list = field(default_factory=list)  # list[CaseResult]
    manifest: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        scored = self.total - self.errored
        return (self.passed / scored) if scored else 0.0


def run_suite(
    cases: list[TestCase],
    provider: Provider,
    *,
    suite_path: Optional[str | Path] = None,
    on_result: Optional[Callable[[CaseResult], None]] = None,
) -> RunSummary:
    """Run every case, score it, and return a `RunSummary`.

    A provider exception on one case is captured as an errored `CaseResult`
    (not fatal) so one flaky call doesn't lose the rest of the run.
    """
    results: list[CaseResult] = []
    for case in cases:
        try:
            comp = provider.complete(case.prompt, system=case.system)
            result = score_case(case, comp)
        except Exception as exc:  # provider/network failure — record, keep going
            result = CaseResult(
                case_id=case.id,
                category=case.category,
                passed=False,
                response="",
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        if on_result:
            on_result(result)

    passed = sum(1 for r in results if r.passed and not r.error)
    errored = sum(1 for r in results if r.error)
    failed = len(results) - passed - errored

    by_category: dict = {}
    for r in results:
        c = by_category.setdefault(r.category, {"total": 0, "passed": 0})
        c["total"] += 1
        if r.passed and not r.error:
            c["passed"] += 1

    manifest = {
        "harness": "blackbox_eval",
        "harness_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "n_cases": len(cases),
    }
    if suite_path is not None:
        manifest["suite_path"] = str(suite_path)
        manifest["suite_sha256"] = suite_sha256(suite_path)

    return RunSummary(
        provider=provider.name,
        model=provider.model,
        total=len(results),
        passed=passed,
        failed=failed,
        errored=errored,
        by_category=by_category,
        results=results,
        manifest=manifest,
    )


def write_results(summary: RunSummary, out_dir: str | Path) -> Path:
    """Write manifest.json, results.jsonl, and summary.json into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(summary.manifest, indent=2), encoding="utf-8")
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in summary.results:
            f.write(json.dumps(r.to_dict()) + "\n")
    summary_doc = {
        "provider": summary.provider,
        "model": summary.model,
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "errored": summary.errored,
        "pass_rate": round(summary.pass_rate, 4),
        "by_category": summary.by_category,
    }
    (out / "summary.json").write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    return out
