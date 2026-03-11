#!/usr/bin/env python3
"""
run_community.py — v1.8 Community Example
==========================================
Loads a 4-bit quantised Llama-3 model via ``unsloth`` (low-end GPU friendly),
wraps it in the ``UniversalHookWrapper``, runs a single generation, and
prints a heartbeat dashboard snapshot.

Usage::

    python run_community.py                   # default prompt
    python run_community.py --prompt "Why is the sky blue?"
    python run_community.py --dual             # enable dual-node mode

Requirements::

    pip install unsloth transformers rich
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.universal_hook import UniversalHookWrapper
from core.dashboard import HeartbeatDashboard


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-bnb-4bit"
DEFAULT_PROMPT = (
    "Explain the ER=EPR correspondence in three sentences."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.8 Community Runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dual", action="store_true", help="Enable dual-node mode")
    args = parser.parse_args()

    print(f"[v1.8] Loading {args.model} …")
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
        flux_ratio=0.25,
        head_rotate_steps=50,
    )

    print(f"[v1.8] Bridge attached: layers {wrapper.mid_idx} → {wrapper.last_idx} "
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
