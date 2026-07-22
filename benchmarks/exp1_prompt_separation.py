#!/usr/bin/env python3
"""benchmarks/exp1_prompt_separation.py — prompt-category separation experiment.

Tests whether the cross-layer zeta metric's cross-sample-null gap is
sensitive to prompt *category*, or whether it is uniform noise regardless of
what kind of prompt produced the sink activation. For each prompt we compute
two null distributions with :func:`length_matched_null_zeta`:

  - ``null_within_category`` : controls = other prompts' sinks from the SAME
                                category
  - ``null_cross_category``  : controls = other prompts' sinks from a
                                DIFFERENT category

If category carries no signal for this metric, the two null gaps should be
indistinguishable. This is a real, graded-by-construction measurement (fixed
canned prompts, real model generations) -- not a synthetic tensor demo (see
``benchmarks/pipeline_demos/``).

Results are written to
  results/runs/<date>_<sha>_<env>/exp1_prompt_separation.json
with a manifest.json (git SHA, pip freeze, device name, seed) beside it.

Usage::

    python -m benchmarks.exp1_prompt_separation --prompts-per-category 8 --model gpt2 --mode passive
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import torch

from benchmarks._harness import set_seed, git_sha, pip_freeze, device_name, env_tag

# ----------------------------------------------------------------------
# Canned prompts, grouped by category. Self-contained (no external dataset
# download) so this experiment has no dependency beyond the model itself.
# ----------------------------------------------------------------------
CATEGORIES: Dict[str, List[str]] = {
    "math": [
        "Question: What is 12 + 37?\nAnswer:",
        "Question: If a train travels 60 miles in 2 hours, what is its average speed?\nAnswer:",
        "Question: What is the derivative of x^2?\nAnswer:",
        "Question: Solve for x: 2x + 5 = 17.\nAnswer:",
        "Question: What is 15% of 200?\nAnswer:",
        "Question: What is the sum of the first 10 positive integers?\nAnswer:",
        "Question: A rectangle has length 8 and width 3. What is its area?\nAnswer:",
        "Question: What is 9 squared?\nAnswer:",
    ],
    "narrative": [
        "Once upon a time, in a quiet village by the sea,",
        "The old lighthouse keeper had not seen another soul in months when",
        "She opened the letter with trembling hands and read:",
        "Deep in the forest, the children found a door that",
        "The spaceship's engines sputtered as the crew realized",
        "He had walked these streets a thousand times, but tonight",
        "The dragon circled once, then landed beside the",
        "In the year the rivers froze early, a stranger arrived in town who",
    ],
    "factual": [
        "Question: What is the capital of France?\nAnswer:",
        "Question: Who wrote the play Romeo and Juliet?\nAnswer:",
        "Question: What is the chemical symbol for gold?\nAnswer:",
        "Question: What year did World War II end?\nAnswer:",
        "Question: What is the largest planet in the solar system?\nAnswer:",
        "Question: How many continents are there on Earth?\nAnswer:",
        "Question: What language is primarily spoken in Brazil?\nAnswer:",
        "Question: What is the boiling point of water in Celsius?\nAnswer:",
    ],
}


def _null_record(csn: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "null_mean": csn["null_mean"],
        "null_std": csn["null_std"],
        "gap": csn["gap"],
        "z_score": csn["z_score"],
        "n_controls": csn["n_controls"],
        "matched_len_tokens": csn["matched_len_tokens"],
        "length_matched": True,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-category separation experiment (EXP1)")
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2).")
    parser.add_argument(
        "--prompts-per-category", type=int, default=8,
        help="Number of prompts to use per category (default: 8).",
    )
    parser.add_argument(
        "--mode", default="passive", choices=["baseline", "passive", "active"],
        help="baseline = bare model; passive = capture-only hooks; active = full bridge.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Max new tokens per completion (default: 32).")
    parser.add_argument(
        "--min-controls", type=int, default=3,
        help="Minimum controls required to record a null result (default: 3).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

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

    # --- Generate + capture activations for every prompt ---
    records: List[Dict[str, Any]] = []
    sources: List[Optional[torch.Tensor]] = []
    sinks: List[Optional[torch.Tensor]] = []
    categories: List[str] = []

    for category, prompts in CATEGORIES.items():
        n = min(args.prompts_per_category, len(prompts))
        for prompt in prompts[:n]:
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

            rec: Dict[str, Any] = {
                "prompt_id": len(records),
                "category": category,
                "prompt": prompt,
                "completion": completion,
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

            categories.append(category)
            records.append(rec)

    # --- Within-category vs cross-category null ---
    # HONESTY-1: uses the length-matched null (see unitarity_labs.core.metrics)
    # so the two conditions are never confounded by sequence-length padding.
    from unitarity_labs.core.metrics import length_matched_null_zeta

    for i, rec in enumerate(records):
        rec["null_within_category"] = None
        rec["null_cross_category"] = None
        if sources[i] is None or sinks[i] is None:
            continue

        within = [
            sinks[j] for j in range(len(sinks))
            if j != i and sinks[j] is not None and categories[j] == categories[i]
        ]
        cross = [
            sinks[j] for j in range(len(sinks))
            if sinks[j] is not None and categories[j] != categories[i]
        ]

        if len(within) >= args.min_controls:
            rec["null_within_category"] = _null_record(
                length_matched_null_zeta(sources[i], sinks[i], within)
            )
        if len(cross) >= args.min_controls:
            rec["null_cross_category"] = _null_record(
                length_matched_null_zeta(sources[i], sinks[i], cross)
            )

    # --- Summary: does the null gap separate by category? ---
    within_gaps = [r["null_within_category"]["gap"] for r in records if r["null_within_category"]]
    cross_gaps = [r["null_cross_category"]["gap"] for r in records if r["null_cross_category"]]

    summary: Dict[str, Any] = {
        "n_prompts": len(records),
        "n_categories": len(CATEGORIES),
        "n_within_computed": len(within_gaps),
        "n_cross_computed": len(cross_gaps),
        "mean_gap_within_category": round(mean(within_gaps), 6) if within_gaps else None,
        "mean_gap_cross_category": round(mean(cross_gaps), 6) if cross_gaps else None,
    }
    if within_gaps and cross_gaps:
        summary["separation"] = round(
            summary["mean_gap_within_category"] - summary["mean_gap_cross_category"], 6
        )
    else:
        summary["separation"] = None

    payload = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "categories": list(CATEGORIES.keys()),
        "prompts_per_category": args.prompts_per_category,
        "n": len(records),
        "summary": summary,
        "results": records,
    }

    # --- Write results + manifest ---
    sha = git_sha()
    run_dir = Path("results/runs") / f"{date.today():%Y%m%d}_{sha}_{env_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "exp1_prompt_separation.json").write_text(
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

    print(f"[exp1_prompt_separation] mode={args.mode} n={len(records)} "
          f"separation={summary['separation']}")
    print(f"[exp1_prompt_separation] results -> {run_dir / 'exp1_prompt_separation.json'}")
    print(f"[exp1_prompt_separation] manifest -> {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
