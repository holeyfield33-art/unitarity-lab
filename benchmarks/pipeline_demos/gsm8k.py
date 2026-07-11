#!/usr/bin/env python3
"""benchmarks/pipeline_demos/gsm8k.py — metric-plumbing DEMO (not an eval).

Feeds synthetic source/sink tensors through the cross-layer cosine metric to
demonstrate the column layout. There is no GSM8K data and no model here — for
a real, graded GSM8K run use ``benchmarks/real_gsm8k.py``.

Usage::

    python -m benchmarks.pipeline_demos.gsm8k --n-problems 5 --seed 42
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit
from benchmarks.pipeline_demos import print_banner


def main() -> None:
    print_banner()
    parser = make_parser("GSM8K metric-plumbing demo (synthetic tensors)")
    parser.add_argument("--n-problems", type=int, default=10,
                        help="Number of synthetic problems (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_problems):
        d = 256
        source = torch.randn(1, 64, d)

        t0 = time.perf_counter()
        sink = source + 0.05 * torch.randn(1, 64, d)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        row = compute_row(source, sink, latency_ms)
        row["problem_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
