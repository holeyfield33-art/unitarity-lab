#!/usr/bin/env python3
"""benchmarks/agent_instruct.py — Agent instruction-following benchmark stub.

Usage::

    python -m benchmarks.agent_instruct --mode passive --seed 42
    python -m benchmarks.agent_instruct --mode active --output results.json
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit


def main() -> None:
    parser = make_parser("Agent instruction-following benchmark")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Number of synthetic tasks (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_tasks):
        d = 384
        source = torch.randn(1, 96, d)
        sink = source + 0.04 * torch.randn(1, 96, d)

        t0 = time.perf_counter()
        time.sleep(0.001)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        accuracy = 1.0 if torch.rand(1).item() > 0.35 else 0.0
        row = compute_row(source, sink, latency_ms, accuracy, seed=args.seed)
        row["task_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
