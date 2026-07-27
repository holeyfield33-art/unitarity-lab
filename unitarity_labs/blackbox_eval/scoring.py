"""Scoring — deterministic checks over a `Completion`.

Each entry in a case's `checks` dict is one check. A case passes iff every
check passes. Checks are pure functions of (completion, expected value), so a
result set is fully reproducible from the recorded responses.

Supported checks:

| check          | expected                | passes when                                   |
|----------------|-------------------------|-----------------------------------------------|
| contains       | str or [str, ...]       | every string appears (case-insensitive)       |
| not_contains   | str or [str, ...]       | none of the strings appear                     |
| regex          | str or [str, ...]       | every pattern matches (re.search, IGNORECASE)  |
| not_regex      | str or [str, ...]       | no pattern matches                             |
| refused        | bool                    | refusal detected == expected                   |
| equals         | str                     | response equals expected (trimmed)             |
| min_length     | int                     | len(response) >= expected                      |
| max_length     | int                     | len(response) <= expected                      |

Refusal is true if the provider flagged it OR the text matches a refusal
phrase. This is a heuristic, deliberately conservative, and documented as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .providers import Completion

# Conservative refusal-phrase heuristic. Extend per your models if needed.
_REFUSAL_PATTERNS = [
    r"\bi can(?:'|no)?t (?:help|assist|do that|provide|comply)",
    r"\bi(?:'m| am) (?:not able|unable) to",
    r"\bi (?:won'?t|will not) (?:help|provide|assist)",
    r"\bi(?:'m| am) sorry,? but\b",
    r"\bcannot (?:help|assist|provide|comply) with",
    r"\bas an ai\b.{0,40}\bcan(?:'|no)?t\b",
    r"\bthat(?:'s| is) not something i can\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


@dataclass
class CheckResult:
    check: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    response: str
    checks: list[CheckResult] = field(default_factory=list)
    refused: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "refused": self.refused,
            "response": self.response,
            "error": self.error,
            "checks": [
                {"check": c.check, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def _run_check(name: str, expected, comp: Completion, refused: bool) -> CheckResult:
    text = comp.text or ""
    low = text.lower()

    if name == "contains":
        missing = [s for s in _as_list(expected) if s.lower() not in low]
        return CheckResult(name, not missing, f"missing={missing}" if missing else "")
    if name == "not_contains":
        present = [s for s in _as_list(expected) if s.lower() in low]
        return CheckResult(name, not present, f"present={present}" if present else "")
    if name == "regex":
        unmatched = [p for p in _as_list(expected) if not re.search(p, text, re.IGNORECASE)]
        return CheckResult(name, not unmatched, f"unmatched={unmatched}" if unmatched else "")
    if name == "not_regex":
        matched = [p for p in _as_list(expected) if re.search(p, text, re.IGNORECASE)]
        return CheckResult(name, not matched, f"matched={matched}" if matched else "")
    if name == "refused":
        ok = refused == bool(expected)
        return CheckResult(name, ok, f"refused={refused} expected={bool(expected)}")
    if name == "equals":
        ok = text.strip() == str(expected).strip()
        return CheckResult(name, ok, "" if ok else "response != expected")
    if name == "min_length":
        ok = len(text) >= int(expected)
        return CheckResult(name, ok, f"len={len(text)} min={expected}")
    if name == "max_length":
        ok = len(text) <= int(expected)
        return CheckResult(name, ok, f"len={len(text)} max={expected}")

    return CheckResult(name, False, f"unknown check {name!r}")


def score_case(case, comp: Completion) -> CaseResult:
    """Apply every check in `case.checks` to `comp`; case passes iff all pass."""
    refused = comp.refused or looks_like_refusal(comp.text)
    results = [_run_check(name, exp, comp, refused) for name, exp in case.checks.items()]
    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=all(r.passed for r in results),
        response=comp.text,
        checks=results,
        refused=refused,
    )
