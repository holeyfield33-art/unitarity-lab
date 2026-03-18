"""
validate_text.py — CLI for extracting and validating spectral metrics from text.
================================================================================
Accepts raw text (from stdin, a file, or a direct argument) containing
model "thoughts" or audit tables, extracts ζ and ⟨r⟩ via regex, evaluates
health against the benchmark, and writes a JSON audit log.

Usage
-----
    # From a string argument:
    python validate_text.py --text "⟨r⟩ = 0.52, ζ = 0.80"

    # From a file:
    python validate_text.py --file audit_table.md

    # From stdin (pipe):
    echo "gap ratio = 0.55, coherence ζ ≈ 0.83" | python validate_text.py

The k = 1 Kar–Berry–Keating Hamiltonian sector is the stabilising
constraint: at k = 1 the level statistics converge to GUE form, fixing the
expected ⟨r⟩ ≈ 0.58 that serves as the benchmark reference.
"""

from __future__ import annotations

import argparse
import json
import sys

from unitarity_labs.core.validator import (
    GROK_4_MARCH_2026_BENCHMARK,
    evaluate_model_health,
    log_audit,
    parse_metrics_from_text,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract spectral metrics from text and validate against benchmark.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text", "-t",
        type=str,
        help="Raw text string containing metrics.",
    )
    source.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file containing metrics.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional label for the audit log filename.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing the JSON audit log.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output the report as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    # Resolve input text.
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide input via --text, --file, or stdin.")
        return 1  # unreachable but keeps type-checkers happy

    # Parse & evaluate.
    stats = parse_metrics_from_text(text)
    report = evaluate_model_health(stats)

    # Output.
    if args.json_output:
        from dataclasses import asdict
        out = {
            "benchmark": GROK_4_MARCH_2026_BENCHMARK,
            "observed": {
                "r_ratio": stats.r_ratio,
                "zeta": stats.zeta,
                "frobenius_stability": stats.frobenius_stability,
            },
            "report": asdict(report),
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print("── Parsed Metrics ──")
        print(f"  ⟨r⟩  = {stats.r_ratio}")
        print(f"  ζ    = {stats.zeta}")
        print(f"  Frob = {stats.frobenius_stability}")
        print()
        print("── Benchmark ──")
        for k, v in GROK_4_MARCH_2026_BENCHMARK.items():
            print(f"  {k} = {v}")
        print()
        print("── Health Report ──")
        print(f"  Spectral Divergence = {report.spectral_divergence:.4f}")
        print(f"  Passed              = {report.passed}")
        print(f"  Details             : {report.details}")

    # Log.
    if not args.no_log:
        path = log_audit(report, stats, tag=args.tag)
        if not args.json_output:
            print(f"\n  Audit log → {path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
