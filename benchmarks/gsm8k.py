#!/usr/bin/env python3
"""benchmarks/gsm8k.py — GSM8K math reasoning benchmark stub.

Evaluates passive vs active modes on GSM8K-style problems.
This is a *harness stub*: it generates synthetic source/sink tensors
to demonstrate the metric pipeline.  Full evaluation requires a
GSM8K dataset and a loaded model.

Usage::

    python -m benchmarks.gsm8k --mode passive --seed 42
    python -m benchmarks.gsm8k --mode active --output results.json
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit


def main() -> None:
    parser = make_parser("GSM8K math reasoning benchmark")
    parser.add_argument("--n-problems", type=int, default=10,
                        help="Number of synthetic problems (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_problems):
        d = 256
        source = torch.randn(1, 64, d)
        sink = source + 0.05 * torch.randn(1, 64, d)

        t0 = time.perf_counter()
        # Placeholder: in a real benchmark, run model.generate() here
        time.sleep(0.001)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        accuracy = 1.0 if torch.rand(1).item() > 0.3 else 0.0
        row = compute_row(source, sink, latency_ms, accuracy, seed=args.seed)
        row["problem_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
