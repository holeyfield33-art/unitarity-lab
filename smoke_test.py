import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper

# 1. Setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- THE FIX ---
# We use mode="active" so the bridge runs the math.
# bridge.enabled stays True so bell_correlation (Zeta) is computed.
hooked_model = UniversalHookWrapper(model, config=model.config, mode="active")
# ---------------

test_prompts = {
    "LOGIC": "The fundamental theorem of arithmetic states that every integer greater than 1 is a prime or a product of primes.",
    "CHAOS": "purple triangle desk jump oxygen 999 elephant record sandwich frequency blue."
}

results = {}

print("--- STARTING ACTIVE-OBSERVER COMPARISON ---")

for label, text in test_prompts.items():
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hooked_model(**inputs)

    # Accessing the internal bell_correlation (Zeta)
    zeta_value = hooked_model.bridge.bell_correlation
    results[label] = zeta_value
    print(f"DEBUG: {label} Captured | Zeta: {zeta_value:.8f}")

print("\n" + "=" * 30)
print("COHERENCE REPORT (SENSOR-ONLY)")
print("=" * 30)
for label, val in results.items():
    print(f"{label}: {val:.8f}")

delta = results["LOGIC"] - results["CHAOS"]
print(f"Structural Delta: {delta:.8f}")
