"""benchmarks/_harness.py — shared benchmark helpers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from core.metrics import manifold_coherence_zeta, baseline_cosine_meanpool, permutation_test_zeta


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create a standard benchmark argument parser."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["passive", "active"], default="active",
                   help="Runtime mode (default: active).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    p.add_argument("--output", type=str, default=None,
                   help="Path to write JSON results (default: stdout).")
    return p


def set_seed(seed: int) -> None:
    """Set global random seed for torch + python random."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_row(
    source: torch.Tensor,
    sink: torch.Tensor,
    latency_ms: float,
    accuracy: float,
    n_perm: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute a single benchmark row with all required columns."""
    zeta = manifold_coherence_zeta(source, sink)
    baseline_cos = baseline_cosine_meanpool(source, sink)
    _, perm_p = permutation_test_zeta(source, sink, n_perm=n_perm, seed=seed)
    return {
        "zeta": round(float(zeta), 6),
        "baseline_cosine": round(float(baseline_cos), 6),
        "permutation_p": round(float(perm_p), 6),
        "latency_ms": round(latency_ms, 2),
        "accuracy": round(accuracy, 4),
    }


def emit(rows: List[Dict[str, Any]], output_path: str | None) -> None:
    """Write benchmark results as JSON."""
    payload = {"results": rows}
    text = json.dumps(payload, indent=2)
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Results written to {output_path}")
    else:
        print(text)
