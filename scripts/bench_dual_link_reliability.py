#!/usr/bin/env python3
"""
bench_dual_link_reliability.py -- Dual-node exchange delivery benchmark
======================================================================
Reproducible proof that the ZeroMQ partner exchange actually delivers after
the reliability fix. A real cross-node run delivered 1 of 800 partner messages
(~0.1%); this benchmark spins up Node A and Node B in a single process (no HF
model needed), performs the startup handshake barrier, then runs N round-robin
send/recv iterations and reports per-node delivery rate.

Pre-fix expectation:  ~0% (slow-joiner drops + 10ms staleness guard).
Post-fix target:      > 90% for BOTH nodes.

Usage::

    python scripts/bench_dual_link_reliability.py
    python scripts/bench_dual_link_reliability.py --iters 500
"""

from __future__ import annotations

import argparse
import sys
import threading

import torch

from unitarity_labs.core.dual_link import DualNodeEntanglementBridge


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-link reliability benchmark")
    parser.add_argument("--port", type=int, default=47555)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    a = DualNodeEntanglementBridge(node_id="A", krylov_dim=args.rank, zmq_port=args.port)
    b = DualNodeEntanglementBridge(node_id="B", krylov_dim=args.rank, zmq_port=args.port)
    try:
        # --- Startup barrier: each side blocks for the other, so run the two
        #     handshakes concurrently. ---
        results: dict = {}

        def _sync(name: str, bridge: DualNodeEntanglementBridge) -> None:
            try:
                results[name] = bridge.synchronize(timeout_ms=10000)
            except Exception as exc:  # noqa: BLE001 - report any failure
                results[name] = exc

        ta = threading.Thread(target=_sync, args=("A", a))
        tb = threading.Thread(target=_sync, args=("B", b))
        ta.start(); tb.start(); ta.join(); tb.join()

        for name in ("A", "B"):
            if isinstance(results.get(name), Exception):
                print(f"[bench] handshake FAILED for {name}: {results[name]!r}")
                return 1
        print(f"[bench] handshake OK: A <-> B, "
              f"epoch_len={results['A'].get('epoch_len')}")

        # --- Round-robin exchange ---
        a_recv = 0
        b_recv = 0
        for _ in range(args.iters):
            basis_a = torch.linalg.qr(torch.randn(args.dim, args.rank))[0]
            basis_b = torch.linalg.qr(torch.randn(args.dim, args.rank))[0]
            a.send_krylov_basis(basis_a)
            b.send_krylov_basis(basis_b)
            if b.recv_partner_basis() is not None:
                b_recv += 1
            if a.recv_partner_basis() is not None:
                a_recv += 1

        a_rate = a_recv / args.iters
        b_rate = b_recv / args.iters
        print(f"[bench] iterations={args.iters}")
        print(f"[bench] Node A delivery: {a_recv}/{args.iters} = {a_rate:.1%}")
        print(f"[bench] Node B delivery: {b_recv}/{args.iters} = {b_rate:.1%}")
        print(f"[bench] A sync_stats: {a.sync_stats}")
        print(f"[bench] B sync_stats: {b.sync_stats}")

        ok = a_rate > args.threshold and b_rate > args.threshold
        print(f"[bench] VERDICT: {'PASS' if ok else 'FAIL'} "
              f"(target >{args.threshold:.0%} both nodes; was ~0.1% pre-fix)")
        return 0 if ok else 1
    finally:
        a.close()
        b.close()


if __name__ == "__main__":
    sys.exit(main())
