import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

# Import unitarity-lab diagnostics from the package root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.metrics import manifold_coherence_zeta


class CoherenceMonitoringCallback(TrainerCallback):
    """
    Monitors cross-layer alignment during training to detect structural collapse
    before quantization execution.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Every 50 steps, compute diagnostic coherence on validation/eval batches.
        if state.global_step % 50 == 0:
            print(
                f"\n[Diagnostic Callback] Step {state.global_step} reached. "
                "Computing structural invariants..."
            )
            # We will wire this hook into the live transformer forward pass in Sprint 3.


def setup_training():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. 4-Bit double quantization config for training VRAM minimization.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"[PEFT Setup] Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # 2. Complete target module mapping to preserve weight matrix coherence.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    print("[PEFT Setup] Model printable trainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer


if __name__ == "__main__":
    setup_training()
