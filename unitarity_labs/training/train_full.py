"""
train_full.py — Sprint 3 full SFT orchestrator.

Trains Qwen2.5-0.5B-Instruct with QLoRA (NF4 double-quant) when a CUDA GPU
is present.  Falls back to float32 CPU loading so the script can be validated
in GPU-less CI/dev-container environments (training will be extremely slow on
CPU; use GPU hardware for real training runs).

TRL 1.7.0 API notes:
  - DataCollatorForCompletionOnlyLM removed → SFTConfig(completion_only_loss=True)
  - SFTConfig uses max_length (not max_seq_length)
"""

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
from trl import SFTConfig, SFTTrainer

# Map package root to access core instrumentation modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.metrics import manifold_coherence_zeta

_HAS_CUDA = torch.cuda.is_available()


class CoherenceMonitoringCallback(TrainerCallback):
    """
    Probes mid-to-last layer activation manifolds during training steps
    using the core 'manifold_coherence_zeta' routine.
    """

    def __init__(self, val_dataset, tokenizer):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Probe structural alignment every 50 steps.
        if state.global_step > 0 and state.global_step % 50 == 0:
            model = kwargs.get("model")
            if model is None:
                return

            model.eval()

            # Pull first validation sample structure.
            sample_messages = self.val_dataset[0]["messages"]
            formatted_input = self.tokenizer.apply_chat_template(
                sample_messages,
                tokenize=True,
                return_tensors="pt",
            ).to(model.device)

            activations = {}

            def get_hook(name):
                def hook(module, input_tensor, output_tensor):
                    if isinstance(output_tensor, tuple):
                        activations[name] = output_tensor[0].detach().cpu()
                    else:
                        activations[name] = output_tensor.detach().cpu()

                return hook

            # Extract underlying layers (unwrapping PEFT & Model structures).
            # Qwen2 structure: PeftModel.base_model (LoraModel)
            #                  → .model (Qwen2ForCausalLM) → .model (Qwen2Model) → .layers
            base_model = model.base_model if hasattr(model, "base_model") else model
            inner = base_model.model if hasattr(base_model, "model") else base_model
            if hasattr(inner, "layers"):
                layers = inner.layers
            elif hasattr(inner, "model") and hasattr(inner.model, "layers"):
                layers = inner.model.layers
            elif hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
                layers = inner.transformer.h
            else:
                raise AttributeError(
                    f"Cannot locate transformer layers in {type(inner).__name__}"
                )

            # Auto-calculate indices: mid-layer and near-end layer.
            mid_idx = len(layers) // 2
            last_idx = len(layers) - 2

            # Register temporary diagnostic hooks.
            h_mid = layers[mid_idx].register_forward_hook(get_hook("mid"))
            h_last = layers[last_idx].register_forward_hook(get_hook("last"))

            with torch.no_grad():
                model(formatted_input)

            # Instantly dismantle hooks to avoid performance loss.
            h_mid.remove()
            h_last.remove()

            # Compute and log zeta similarity.
            if "mid" in activations and "last" in activations:
                zeta = manifold_coherence_zeta(activations["mid"], activations["last"])
                print(
                    f"\n[Step {state.global_step} Monitor] "
                    f"Manifold Coherence (Zeta): {zeta:.6f}"
                )

            model.train()


def run_sft_training():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Quantization Configuration (Double Quantized NF4) — CUDA only.
    if _HAS_CUDA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        device_map = "auto"
        use_fp16 = True
        use_cpu = False
        print("[Training] CUDA detected — loading with NF4 double-quant.")
    else:
        bnb_config = None
        device_map = "cpu"
        use_fp16 = False
        use_cpu = True
        print(
            "[Training] WARNING: No CUDA GPU detected. "
            "Loading model in float32 on CPU (training will be slow; "
            "use CUDA hardware for production runs)."
        )

    print(f"[Training] Loading tokenizer & model for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if _HAS_CUDA:
        model = prepare_model_for_kbit_training(model)

    # 2. Configure PEFT target projection modules.
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
    model.print_trainable_parameters()

    # 3. Load programmatically generated datapacks.
    datapack_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "datapack"
    )
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(datapack_root, "train.jsonl"),
            "validation": os.path.join(datapack_root, "val.jsonl"),
        },
    )

    # 4. Apply chat template — map conversations to flat text strings.
    def format_prompts(batch):
        formatted_list = []
        for conversation in batch["messages"]:
            text = tokenizer.apply_chat_template(conversation, tokenize=False)
            formatted_list.append(text)
        return {"text": formatted_list}

    processed_dataset = dataset.map(format_prompts, batched=True)

    # 5. Define Training Args.
    # DataCollatorForCompletionOnlyLM was removed in TRL 1.7.0;
    # completion_only_loss=True in SFTConfig is the replacement.
    # warmup_ratio removed in transformers v5.2 — compute warmup_steps explicitly.
    # 1800 samples / (batch 4 * grad_accum 4) = 112 optimizer steps per epoch.
    warmup_steps = max(1, int(112 * 0.03))  # ≈3 steps
    training_args = SFTConfig(
        output_dir="./lora_checkpoints",
        dataset_text_field="text",
        max_length=512,                     # replaces max_seq_length (TRL 1.7.0)
        completion_only_loss=True,          # replaces DataCollatorForCompletionOnlyLM
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=use_fp16,
        use_cpu=use_cpu,
        report_to="none",
    )

    # Instantiate Trainer with Manifold Coherence monitoring callback.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        callbacks=[
            CoherenceMonitoringCallback(
                processed_dataset["validation"], tokenizer
            )
        ],
    )

    print("[Training] Launching SFT execution run...")
    trainer.train()

    # Save trained parameters.
    print("[Training] Completed. Saving LoRA adapters to ./final_adapters...")
    model.save_pretrained("./final_adapters")
    tokenizer.save_pretrained("./final_adapters")


if __name__ == "__main__":
    run_sft_training()
