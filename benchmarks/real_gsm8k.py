#!/usr/bin/env python3
"""benchmarks/real_gsm8k.py — a REAL, graded GSM8K evaluation.

Unlike the ``pipeline_demos/`` scripts (synthetic tensors, no grading), this
loads GSM8K, generates answers with an actual model, and grades each problem by
numeric extraction of the final answer. Every recorded number is measured — no
``torch.rand`` accuracy, no ``time.sleep`` latency, no ``permutation_test_zeta``.

Per problem it records:
  - correct (bool)      : graded numeric match against the gold answer
  - zeta_raw            : pre-intervention cross-layer cosine (None in baseline)
  - cross_sample_null   : {null_mean,null_std,gap,z_score} with the OTHER
                          problems' sink activations as controls (None if the
                          mode captures no sink, or <3 controls)
  - spectral_gap        : source-layer spectral gap (None in baseline)
  - latency_ms          : measured wall-clock generation latency
  - prompt_tokens / completion_tokens : real token counts

Results are written to
  results/runs/<date>_<sha>_<env>/gsm8k_real.json
with a manifest.json (git SHA, pip freeze, device name, seed) beside it.

Usage::

    python -m benchmarks.real_gsm8k --n 5 --model gpt2 --mode passive
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import torch

from benchmarks._harness import git_sha, pip_freeze, device_name, env_tag


# ----------------------------------------------------------------------
# Answer grading
# ----------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> Optional[float]:
    """Extract the last numeric token from a string (commas stripped)."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _gold_number(answer: str) -> Optional[float]:
    """GSM8K gold answers put the final value after '#### '."""
    if "####" in answer:
        return _last_number(answer.split("####")[-1])
    return _last_number(answer)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Real, graded GSM8K evaluation")
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2).")
    parser.add_argument("--n", type=int, default=100, help="Number of problems (default: 100).")
    parser.add_argument(
        "--mode", default="passive", choices=["baseline", "passive", "active"],
        help="baseline = bare model; passive = capture-only hooks; active = full bridge.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max new tokens per answer (default: 256).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- Load dataset ---
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "real_gsm8k requires the 'datasets' package: pip install datasets"
        ) from e

    # "openai/gsm8k" is the canonical repo id; fall back to the legacy short
    # name for older `datasets` versions.
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")
    n = min(args.n, len(ds))

    # --- Load model ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    wrapper = None
    if args.mode in ("passive", "active"):
        from unitarity_labs.core.universal_hook import UniversalHookWrapper
        wrapper = UniversalHookWrapper(
            model=model, config=model.config, mode=args.mode,
        )

    # --- Evaluate ---
    records = []
    # Cache per-problem source/sink for the cross-sample null (controls = other
    # problems' sink activations).
    sources: List[Optional[torch.Tensor]] = []
    sinks: List[Optional[torch.Tensor]] = []

    for i in range(n):
        question = ds[i]["question"]
        gold = _gold_number(ds[i]["answer"])
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        completion_ids = out_ids[0][prompt_tokens:]
        completion_tokens = int(completion_ids.shape[-1])
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        pred = _last_number(completion)
        correct = (
            gold is not None and pred is not None and abs(pred - gold) < 1e-6
        )

        rec = {
            "problem_id": i,
            "correct": bool(correct),
            "gold": gold,
            "pred": pred,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        if wrapper is not None:
            rec["zeta_raw"] = wrapper.bridge.raw_sink_zeta
            rec["spectral_gap"] = wrapper.bridge.spectral_gap()
            src = wrapper.bridge._source_activation
            snk = wrapper.bridge._sink_activation
            sources.append(src.detach().to("cpu") if src is not None else None)
            sinks.append(snk.detach().to("cpu") if snk is not None else None)
        else:
            rec["zeta_raw"] = None
            rec["spectral_gap"] = None
            sources.append(None)
            sinks.append(None)

        records.append(rec)

    # --- Cross-sample null: matched (source_i, sink_i) vs (source_i, sink_j!=i) ---
    # HONESTY-1: uses the length-matched null. The old padded null deflated
    # null_std (padding caps cosine at ||truncated||/||full||) and inflated
    # z_score; every pair below is compared at a common truncated length.
    from unitarity_labs.core.metrics import length_matched_null_zeta

    for i, rec in enumerate(records):
        rec["cross_sample_null"] = None
        if sources[i] is None or sinks[i] is None:
            continue
        controls = [sinks[j] for j in range(len(sinks)) if j != i and sinks[j] is not None]
        if len(controls) < 3:
            continue
        csn = length_matched_null_zeta(sources[i], sinks[i], controls)
        rec["cross_sample_null"] = {
            "null_mean": csn["null_mean"],
            "null_std": csn["null_std"],
            "gap": csn["gap"],
            "z_score": csn["z_score"],
            "matched_len_tokens": csn["matched_len_tokens"],
            "length_matched": True,
        }

    accuracy = sum(1 for r in records if r["correct"]) / max(len(records), 1)

    payload = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "n": len(records),
        "accuracy": round(accuracy, 6),
        "results": records,
    }

    # --- Write results + manifest ---
    sha = git_sha()
    run_dir = Path("results/runs") / f"{date.today():%Y%m%d}_{sha}_{env_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "gsm8k_real.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    manifest = {
        "git_sha": sha,
        "device": device_name(),
        "seed": args.seed,
        "model": args.model,
        "mode": args.mode,
        "n": len(records),
        "pip_freeze": pip_freeze(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[real_gsm8k] mode={args.mode} n={len(records)} "
          f"accuracy={accuracy:.4f}")
    print(f"[real_gsm8k] results -> {run_dir / 'gsm8k_real.json'}")
    print(f"[real_gsm8k] manifest -> {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
