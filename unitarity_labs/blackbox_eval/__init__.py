"""blackbox_eval — behavioral (black-box) testing for any language model.

This is the complement to the rest of `unitarity_labs`. The spectral/hidden-state
tooling elsewhere is **white-box**: it reads a model's internal activations, so it
only works on small open-weight models you run yourself. This package is
**black-box**: prompt in, text out, scored by rules. It therefore works on *any*
model reachable behind a `Provider` — including large hosted API models
(Anthropic, OpenAI) that expose no internals.

Pieces:
- `providers`  — a `Provider` protocol plus Mock / Anthropic / OpenAI / local-HF
  implementations. SDK imports are lazy, so importing this package pulls in
  nothing beyond the stdlib.
- `suite`      — `TestCase` and a JSON/JSONL suite loader.
- `scoring`    — deterministic checks (contains, regex, refusal, length, ...).
- `runner`     — runs a suite against a provider and writes results + a
  reproducibility manifest (provider, model, suite hash, timestamp, versions).

No numbers ship pre-computed here: results come only from running a real suite
against a real provider. The bundled tests exercise everything through
`MockProvider`, so the suite is green with no network and no API keys.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .providers import (
    Provider,
    Completion,
    MockProvider,
    AnthropicProvider,
    OpenAIProvider,
    LocalTransformersProvider,
    get_provider,
)
from .suite import TestCase, load_suite, suite_sha256
from .scoring import score_case, CheckResult, CaseResult
from .runner import run_suite, RunSummary

__all__ = [
    "__version__",
    "Provider",
    "Completion",
    "MockProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalTransformersProvider",
    "get_provider",
    "TestCase",
    "load_suite",
    "suite_sha256",
    "score_case",
    "CheckResult",
    "CaseResult",
    "run_suite",
    "RunSummary",
]
