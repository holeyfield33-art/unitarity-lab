#!/usr/bin/env python3
"""benchmarks/pipeline_demos/agent_instruct.py — metric-plumbing DEMO.

Synthetic tensors only — no tasks, no instruction following, no grading.
Demonstrates the metric column layout. Not an evaluation.

Usage::

    python -m benchmarks.pipeline_demos.agent_instruct --n-tasks 5 --seed 42
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit
from benchmarks.pipeline_demos import print_banner


def main() -> None:
    print_banner()
    parser = make_parser("Agent-instruct metric-plumbing demo (synthetic tensors)")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Number of synthetic tasks (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_tasks):
        d = 384
        source = torch.randn(1, 96, d)

        t0 = time.perf_counter()
        sink = source + 0.04 * torch.randn(1, 96, d)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        row = compute_row(source, sink, latency_ms)
        row["task_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
