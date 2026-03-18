#!/usr/bin/env python3
"""
start_node.py — Node entry point (v3.0.0-Singularity)
=======================================================
Auto-detecting entry point for the unitarity-lab runtime.

Detects your hardware (CPU-only, laptop GPU, prosumer GPU, server GPU),
selects the appropriate ``PrecisionClass``, and boots a
``UniversalHookWrapper``-instrumented model with the full v3.0 stack.

Modes
-----
  --mode-passive   Hooks capture metrics only; no tensor mutation.
  --mode-active    Full bridge intervention (default).

Usage::

    python start_node.py                       # auto-detect, active mode
    python start_node.py --mode-passive        # passive (metrics only)
    python start_node.py --node-id B           # join as Node B
    python start_node.py --precision BF16      # force precision class
    python start_node.py --dual                # enable dual-node mode
    python start_node.py --model <hf_model_id> # custom model
"""

from __future__ import annotations

import argparse
import sys

import torch

from unitarity_labs.core.precision_projector import PrecisionClass
from unitarity_labs.core.universal_hook import UniversalHookWrapper
from unitarity_labs.core.version import __version__


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
        # T4 and similar Turing GPUs (compute capability < 8.0) lack
        # native BF16 ALUs; route them through the INT4/FP16 path.
        if props.major < 8:
            print(f"[Ghost] 8–24 GB VRAM but compute capability "
                  f"{props.major}.{props.minor} < 8.0 → INT4 (FP16 quant)")
            return PrecisionClass.INT4
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


def _input_device(model) -> torch.device:
    """Infer the device inputs should be placed on.

    With ``device_map='auto'`` the embedding table may be on a
    different device than the decoder layers.  HF's generate()
    dispatches through the embed layer first, so we take the
    device of the embed_tokens (or first parameter as fallback).
    """
    for name in ('model.embed_tokens', 'transformer.wte'):
        parts = name.split('.')
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            return next(obj.parameters()).device
        except (AttributeError, StopIteration):
            continue
    # Fallback: first parameter
    return next(model.parameters()).device


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
        description=f"unitarity-lab {__version__} — Start a runtime node",
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
        help="Enable dual-node ZMQ coordination.",
    )
    parser.add_argument(
        "--mode-passive", dest="mode", action="store_const", const="passive",
        default="active",
        help="Passive mode: hooks capture metrics only, no tensor mutation.",
    )
    parser.add_argument(
        "--mode-active", dest="mode", action="store_const", const="active",
        help="Active mode: full bridge intervention (default).",
    )
    parser.add_argument(
        "--min-compute-tps", type=float, default=12.0,
        help="Minimum tokens/s for compute-tier classification (default: 12.0).",
    )
    parser.add_argument(
        "--epoch-len", type=int, default=16,
        help="Initial gossip epoch length in tokens (default: 16).",
    )
    parser.add_argument(
        "--prompt", default="Explain cross-layer alignment in three sentences.",
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

    # --- Wrap with unitarity-lab v3.0.0 ---
    print(f"[Node] unitarity-lab {__version__}")
    print(f"[Node] mode={args.mode}, node_id={args.node_id}, "
          f"precision={precision.value}, epoch_len={args.epoch_len}, "
          f"dual={args.dual}, min_compute_tps={args.min_compute_tps}")

    wrapper = UniversalHookWrapper(
        model=model,
        config=model.config,
        node_id=args.node_id,
        enable_dual=args.dual,
        mode=args.mode,
        precision=precision,
        initial_epoch_len=args.epoch_len,
        reorth_interval=256,
    )

    print(f"[Node] Bridge: layers {wrapper.mid_idx} → {wrapper.last_idx} "
          f"({wrapper.num_layers} total), "
          f"{int(wrapper.head_mask.sum())}/{wrapper.num_heads} heads active")

    # --- Generate ---
    print(f"\n[Node] Generating with prompt: {args.prompt!r}\n")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(_input_device(model)) for k, v in inputs.items()}

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
    print(f"\n[Node] Metrics after generation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- Optional dashboard ---
    if args.dashboard:
        from unitarity_labs.core.dashboard import HeartbeatDashboard
        dash = HeartbeatDashboard(wrapper)
        dash.run_once()

    print(f"\n[Node] Session complete. {__version__}")


if __name__ == "__main__":
    main()
