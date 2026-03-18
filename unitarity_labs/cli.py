"""
cli.py — Package-internal CLI entry points for unitarity-labs.
===============================================================
These functions are referenced by pyproject.toml [project.scripts]
so that ``pip install unitarity-labs`` creates working console scripts
without requiring root-level .py files in the wheel.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch

from unitarity_labs.core.precision_projector import PrecisionClass
from unitarity_labs.core.version import __version__


# ── Hardware detection (ported from start_node.py) ────────────────────

def _detect_precision() -> PrecisionClass:
    """Auto-detect the best PrecisionClass for the current hardware."""
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
        print("[Ghost] ≥ 24 GB VRAM → BF16 (server-class)")
        return PrecisionClass.BF16


def _detect_torch_dtype(precision: PrecisionClass) -> torch.dtype:
    """Map PrecisionClass to the torch dtype used for model loading."""
    return {
        PrecisionClass.INT4: torch.float16,
        PrecisionClass.FP8_E4M3: torch.float16,
        PrecisionClass.BF16: torch.bfloat16,
        PrecisionClass.FP32: torch.float32,
    }[precision]


_DEFAULT_MODELS = {
    PrecisionClass.INT4: "unsloth/Llama-3.2-1B-bnb-4bit",
    PrecisionClass.FP8_E4M3: "unsloth/Llama-3.2-1B-bnb-4bit",
    PrecisionClass.BF16: "meta-llama/Llama-3.2-1B",
    PrecisionClass.FP32: "meta-llama/Llama-3.2-1B",
}


# ── Main entry point ─────────────────────────────────────────────────

def main() -> int:
    """Entry point for the ``unitarity-start`` console script."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unitarity_labs.core.universal_hook import UniversalHookWrapper

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
        precision = _detect_precision()

    # --- Select model ---
    model_id = args.model or _DEFAULT_MODELS.get(
        precision, _DEFAULT_MODELS[PrecisionClass.BF16]
    )

    # --- Load ---
    dtype = _detect_torch_dtype(precision)
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

    # --- Wrap with unitarity-lab ---
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
    print("\n[Node] Metrics after generation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- Optional dashboard ---
    if args.dashboard:
        from unitarity_labs.core.dashboard import HeartbeatDashboard
        dash = HeartbeatDashboard(wrapper)
        dash.run_once()

    print(f"\n[Node] Session complete. {__version__}")
    return 0


# ── Validate-text entry point ────────────────────────────────────────

def validate_text_main() -> int:
    """Entry point for the ``unitarity-validate`` console script."""
    import json
    from unitarity_labs.core.validator import (
        GROK_4_MARCH_2026_BENCHMARK,
        evaluate_model_health,
        log_audit,
        parse_metrics_from_text,
    )

    parser = argparse.ArgumentParser(
        description="Extract spectral metrics from text and validate against benchmark.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", "-t", type=str, help="Raw text string containing metrics.")
    source.add_argument("--file", "-f", type=str, help="Path to a text file containing metrics.")
    parser.add_argument("--tag", type=str, default="", help="Optional label for the audit log filename.")
    parser.add_argument("--no-log", action="store_true", help="Skip writing the JSON audit log.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output the report as JSON instead of human-readable text.")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide input via --text, --file, or stdin.")
        return 1

    stats = parse_metrics_from_text(text)
    report = evaluate_model_health(stats)

    if args.json_output:
        from dataclasses import asdict
        out = {
            "benchmark": GROK_4_MARCH_2026_BENCHMARK,
            "observed": {
                "r_ratio": stats.r_ratio,
                "zeta": stats.zeta,
                "frobenius_stability": stats.frobenius_stability,
            },
            "report": asdict(report),
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print("── Parsed Metrics ──")
        print(f"  ⟨r⟩  = {stats.r_ratio}")
        print(f"  ζ    = {stats.zeta}")
        print(f"  Frob = {stats.frobenius_stability}")
        print()
        print("── Health Report ──")
        print(f"  Spectral Divergence = {report.spectral_divergence:.4f}")
        print(f"  Passed              = {report.passed}")
        print(f"  Details             : {report.details}")

    if not args.no_log:
        path = log_audit(report, stats, tag=args.tag)
        if not args.json_output:
            print(f"\n  Audit log → {path}")

    return 0 if report.passed else 1
