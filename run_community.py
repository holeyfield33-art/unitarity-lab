#!/usr/bin/env python3
"""
run_community.py — Community Example (v3.0.0-Singularity)
==========================================================
Loads a 4-bit quantised Llama-3 model via ``unsloth`` (low-end GPU friendly),
wraps it in the ``UniversalHookWrapper``, runs a single generation, and
prints a heartbeat dashboard snapshot.

Usage::

    python run_community.py                        # default prompt, active mode
    python run_community.py --mode-passive         # passive (metrics only)
    python run_community.py --prompt "Why is the sky blue?"
    python run_community.py --dual                 # enable dual-node mode

Requirements::

    pip install unsloth transformers rich
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unitarity_labs.core.universal_hook import UniversalHookWrapper
from unitarity_labs.core.dashboard import HeartbeatDashboard
from unitarity_labs.core.version import __version__


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-bnb-4bit"
DEFAULT_PROMPT = "Explain cross-layer alignment in three sentences."


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"unitarity-lab {__version__} — Community Runner",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dual", action="store_true", help="Enable dual-node mode")
    parser.add_argument(
        "--mode-passive", dest="mode", action="store_const", const="passive",
        default="active",
        help="Passive mode: hooks capture metrics only, no tensor mutation.",
    )
    parser.add_argument(
        "--mode-active", dest="mode", action="store_const", const="active",
        help="Active mode: full bridge intervention (default).",
    )
    args = parser.parse_args()

    print(f"[Node] unitarity-lab {__version__}")
    print(f"[Node] Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    wrapper = UniversalHookWrapper(
        model=model,
        config=model.config,
        enable_dual=args.dual,
        mode=args.mode,
        flux_ratio=0.25,
        head_rotate_steps=50,
    )

    print(f"[Node] mode={args.mode}, Bridge: layers {wrapper.mid_idx} → {wrapper.last_idx} "
          f"({wrapper.num_layers} total), {int(wrapper.head_mask.sum())}/"
          f"{wrapper.num_heads} heads active")

    # Generate
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = wrapper(**inputs, max_new_tokens=args.max_new_tokens).logits.argmax(dim=-1)

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(generated)
    print(f"{'='*60}\n")

    # Dashboard snapshot
    dash = HeartbeatDashboard(wrapper)
    dash.run_once()


if __name__ == "__main__":
    main()
