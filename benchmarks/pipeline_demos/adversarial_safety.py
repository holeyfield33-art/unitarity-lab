#!/usr/bin/env python3
"""benchmarks/pipeline_demos/adversarial_safety.py — metric-plumbing DEMO.

Synthetic tensors only — no prompts, no model outputs, no safety judgement.
The larger perturbation here just yields a lower cross-layer cosine; it is a
plumbing illustration, not a safety evaluation.

Usage::

    python -m benchmarks.pipeline_demos.adversarial_safety --n-prompts 5 --seed 42
"""

from __future__ import annotations

import time

import torch

from benchmarks._harness import make_parser, set_seed, compute_row, emit
from benchmarks.pipeline_demos import print_banner


def main() -> None:
    print_banner()
    parser = make_parser("Adversarial-safety metric-plumbing demo (synthetic tensors)")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Number of synthetic prompts (default: 10).")
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for i in range(args.n_prompts):
        d = 256
        source = torch.randn(1, 48, d)

        t0 = time.perf_counter()
        # Larger perturbation -> lower cosine (illustrative only).
        sink = source + 0.15 * torch.randn(1, 48, d)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        row = compute_row(source, sink, latency_ms)
        row["prompt_id"] = i
        row["mode"] = args.mode
        rows.append(row)

    emit(rows, args.output)


if __name__ == "__main__":
    main()
