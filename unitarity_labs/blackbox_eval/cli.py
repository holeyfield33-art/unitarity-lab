"""CLI: python -m unitarity_labs.blackbox_eval run ...

Examples:
    # Offline dry run against the bundled starter suite (no keys, no network):
    python -m unitarity_labs.blackbox_eval run --provider mock

    # Test a big hosted model (needs `pip install anthropic` + ANTHROPIC_API_KEY):
    python -m unitarity_labs.blackbox_eval run --provider anthropic --model claude-opus-5

    # Test an OpenAI model (needs `pip install openai` + OPENAI_API_KEY):
    python -m unitarity_labs.blackbox_eval run --provider openai --model gpt-4o-mini

    # Custom suite + write results:
    python -m unitarity_labs.blackbox_eval run --provider anthropic \\
        --suite my_suite.jsonl --out results/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .providers import get_provider
from .runner import run_suite, write_results
from .scoring import CaseResult
from .suite import load_suite

_DEFAULT_SUITE = Path(__file__).parent / "suites" / "starter.jsonl"


def _print_result(r: CaseResult) -> None:
    mark = "PASS" if (r.passed and not r.error) else ("ERR " if r.error else "FAIL")
    line = f"  [{mark}] {r.case_id} ({r.category})"
    if r.error:
        line += f"  {r.error}"
    elif not r.passed:
        fails = [c.check for c in r.checks if not c.passed]
        line += f"  failed: {', '.join(fails)}"
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blackbox_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite against a provider")
    run.add_argument("--provider", default="mock", help="mock|anthropic|openai|local")
    run.add_argument("--model", default=None, help="model id (provider default if omitted)")
    run.add_argument("--suite", default=str(_DEFAULT_SUITE), help="path to a .json/.jsonl suite")
    run.add_argument("--out", default=None, help="directory to write manifest/results/summary")
    run.add_argument("--max-tokens", type=int, default=1024)

    args = parser.parse_args(argv)

    if args.command == "run":
        suite_path = args.suite
        cases = load_suite(suite_path)
        kwargs = {}
        if args.provider in ("anthropic", "openai"):
            kwargs["max_tokens"] = args.max_tokens
        provider = get_provider(args.provider, model=args.model, **kwargs)

        print(f"Running {len(cases)} cases against {provider.name}:{provider.model}\n")
        summary = run_suite(cases, provider, suite_path=suite_path, on_result=_print_result)

        print(
            f"\n{summary.passed} passed, {summary.failed} failed, "
            f"{summary.errored} errored  (pass rate {summary.pass_rate:.0%} of scored)"
        )
        for cat, c in sorted(summary.by_category.items()):
            print(f"    {cat}: {c['passed']}/{c['total']}")

        if args.out:
            out = write_results(summary, args.out)
            print(f"\nWrote manifest/results/summary to {out}")

        # Non-zero exit if anything failed or errored, for CI use.
        return 0 if (summary.failed == 0 and summary.errored == 0) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
