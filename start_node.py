#!/usr/bin/env python3
"""
start_node.py — The Ghost Script (v2.0)
=========================================
Auto-detecting entry point for the Holey-Field Network.

Detects your hardware (CPU-only, laptop GPU, prosumer GPU, server GPU),
selects the appropriate ``PrecisionClass``, and boots a
``UniversalHookWrapper``-instrumented model with the full v2.0 stack:

  - Byzantine Kill-Switch
  - Precision Alignment (DequantAdapter + dithering)
  - Adaptive Gossip Epoch
  - Periodic Re-orthogonalization

Usage::

    python start_node.py                       # auto-detect everything
    python start_node.py --node-id B           # join as Node B
    python start_node.py --precision BF16      # force precision class
    python start_node.py --dual                # enable dual-node mode
    python start_node.py --model <hf_model_id> # custom model
"""

from __future__ import annotations

import argparse
import sys

import torch

from core.precision_projector import PrecisionClass
from core.universal_hook import UniversalHookWrapper


# ======================================================================
# Hardware Detection
# ======================================================================

def detect_precision() -> PrecisionClass:
    """Auto-detect the best PrecisionClass for the current hardware.

    Returns
    -------
    PrecisionClass
        INT4  — GPU with < 8 GB VRAM  (laptop-class, needs quantisation)
        BF16  — GPU with 8–24 GB VRAM (prosumer, native BF16 support)
        FP32  — No GPU / CPU-only fallback
        BF16  — GPU with >= 24 GB (server, BF16 is efficient enough)
    """
    if not torch.cuda.is_available():
        print("[Ghost] No CUDA GPU detected → FP32 (CPU mode)")
        return PrecisionClass.FP32

    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_mem / (1024 ** 3)
        name = props.name
    except Exception:
        print("[Ghost] Could not query GPU properties → FP32 fallback")
        return PrecisionClass.FP32

    print(f"[Ghost] Detected GPU: {name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 8:
        print("[Ghost] < 8 GB VRAM → INT4 (quantised, laptop-class)")
        return PrecisionClass.INT4
    elif vram_gb < 24:
        print("[Ghost] 8–24 GB VRAM → BF16 (prosumer)")
        return PrecisionClass.BF16
    else:
        print(f"[Ghost] ≥ 24 GB VRAM → BF16 (server-class)")
        return PrecisionClass.BF16


def detect_torch_dtype(precision: PrecisionClass) -> torch.dtype:
    """Map PrecisionClass to the torch dtype used for model loading."""
    return {
        PrecisionClass.INT4: torch.float16,     # 4-bit quant handled by bnb
        PrecisionClass.FP8_E4M3: torch.float16,
        PrecisionClass.BF16: torch.bfloat16,
        PrecisionClass.FP32: torch.float32,
    }[precision]


# ======================================================================
# Model Loading
# ======================================================================

DEFAULT_MODELS = {
    PrecisionClass.INT4: "unsloth/Llama-3.2-1B-bnb-4bit",
    PrecisionClass.FP8_E4M3: "unsloth/Llama-3.2-1B-bnb-4bit",
    PrecisionClass.BF16: "meta-llama/Llama-3.2-1B",
    PrecisionClass.FP32: "meta-llama/Llama-3.2-1B",
}


def load_model(model_id: str, precision: PrecisionClass):
    """Load a HuggingFace causal LM with the appropriate dtype/device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = detect_torch_dtype(precision)
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    print(f"[Ghost] Loading model: {model_id}")
    print(f"[Ghost] dtype={dtype}, device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    return model, tokenizer


# ======================================================================
# Main Entry Point
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holeyfield v2.0 — Start a node on the Holey-Field Network",
    )
    parser.add_argument(
        "--node-id", default="A", choices=["A", "B"],
        help="Node identity: A (relay) or B (peer). Default: A",
    )
    parser.add_argument(
        "--precision", default=None,
        choices=[pc.value for pc in PrecisionClass],
        help="Force a PrecisionClass (auto-detected if omitted).",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model ID (auto-selected if omitted).",
    )
    parser.add_argument(
        "--dual", action="store_true",
        help="Enable dual-node ZMQ entanglement.",
    )
    parser.add_argument(
        "--epoch-len", type=int, default=16,
        help="Initial gossip epoch length in tokens (default: 16).",
    )
    parser.add_argument(
        "--prompt", default="Explain the ER=EPR correspondence in three sentences.",
        help="Generation prompt.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Show the rich heartbeat dashboard after generation.",
    )
    args = parser.parse_args()

    # --- Detect or override precision ---
    if args.precision:
        precision = PrecisionClass(args.precision)
        print(f"[Ghost] Precision forced to {precision.value}")
    else:
        precision = detect_precision()

    # --- Select model ---
    model_id = args.model or DEFAULT_MODELS.get(precision, DEFAULT_MODELS[PrecisionClass.BF16])

    # --- Load ---
    model, tokenizer = load_model(model_id, precision)

    # --- Wrap with Holeyfield v2.0 ---
    print(f"[Ghost] Wrapping with UniversalHookWrapper (v2.0)")
    print(f"[Ghost]   node_id={args.node_id}, precision={precision.value}, "
          f"epoch_len={args.epoch_len}, dual={args.dual}")

    wrapper = UniversalHookWrapper(
        model=model,
        config=model.config,
        node_id=args.node_id,
        enable_dual=args.dual,
        precision=precision,
        initial_epoch_len=args.epoch_len,
        reorth_interval=256,
    )

    print(f"[Ghost] Bridge: layers {wrapper.mid_idx} → {wrapper.last_idx} "
          f"({wrapper.num_layers} total), "
          f"{int(wrapper.head_mask.sum())}/{wrapper.num_heads} heads active")

    # --- Generate ---
    print(f"\n[Ghost] Generating with prompt: {args.prompt!r}\n")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("=" * 60)
    print(generated)
    print("=" * 60)

    # --- Metrics ---
    metrics = wrapper.get_metrics()
    print(f"\n[Ghost] Metrics after generation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- Optional dashboard ---
    if args.dashboard:
        from core.dashboard import HeartbeatDashboard
        dash = HeartbeatDashboard(wrapper)
        dash.run_once()

    print("\n[Ghost] Node session complete. v2.0-stable sealed.")


if __name__ == "__main__":
    main()
