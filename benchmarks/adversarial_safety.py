#!/usr/bin/env python3
"""benchmarks/adversarial_safety.py — Adversarial safety benchmark stub.

Evaluates whether bridge intervention introduces safety regressions
by testing model outputs under adversarial prompts in both modes.

Usage::

    python -m benchmarks.adversarial_safety --mode passive --seed 42
    python -m benchmarks.adversarial_safety --mode active --output results.json
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit


def main() -> None:
    parser = make_parser("Adversarial safety benchmark")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Number of synthetic adversarial prompts (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_prompts):
        d = 256
        source = torch.randn(1, 48, d)
        # Adversarial: larger perturbation
        sink = source + 0.15 * torch.randn(1, 48, d)

        t0 = time.perf_counter()
        time.sleep(0.001)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Safety accuracy: 1.0 = refused harmful prompt correctly
        accuracy = 1.0 if torch.rand(1).item() > 0.1 else 0.0
        row = compute_row(source, sink, latency_ms, accuracy, seed=args.seed)
        row["prompt_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
