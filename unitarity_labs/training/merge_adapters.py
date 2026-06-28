# merge_adapters.py
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_dir = "./final_adapters"
    output_dir = "./merged_model"

    print("[Merger] Loading base model in unquantized float16...")
    # Loading on CPU is safe; requires ~1.1 GB RAM for 0.5B model.
    # transformers >= 5.0 uses `dtype` instead of the deprecated `torch_dtype`.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    if not os.path.exists(adapter_dir):
        print(f"[Merger] ERROR: Adapter directory '{adapter_dir}' not found.")
        print("[Merger] Please ensure final SFT training completed on your GPU machine.")
        sys.exit(1)

    print(f"[Merger] Wrapping model with adapters from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("[Merger] Executing in-place weight matrix merge...")
    merged_model = model.merge_and_unload()

    print(f"[Merger] Writing merged FP16 weights to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("[Merger] Merge complete.")


if __name__ == "__main__":
    merge_and_save()
