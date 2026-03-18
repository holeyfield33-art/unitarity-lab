#!/usr/bin/env python3
"""
Test script to verify active mode works correctly on GPU.
Run this after applying fixes.
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper


def test_active_mode():
    """Test active mode functionality."""
    print("=" * 60)
    print("TESTING ACTIVE MODE DEVICE PLACEMENT")
    print("=" * 60)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if cuda_available else "cpu"
    )

    # Create wrapper in ACTIVE mode
    print("\nCreating UniversalHookWrapper (ACTIVE mode)...")
    wrapper = UniversalHookWrapper(
        model=model,
        config=model.config,
        mode="active",
        flux_ratio=0.25
    )

    # Force device sync
    print("Ensuring device placement...")
    wrapper.ensure_device()

    # Test forward pass
    print("\nRunning forward pass...")
    prompt = "The geometric nature of consciousness is"
    inputs = tokenizer(prompt, return_tensors="pt")

    if cuda_available:
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    try:
        with torch.no_grad():
            wrapper(**inputs)
        print("PASS: Forward pass successful")
    except Exception as e:
        print(f"FAIL: Forward pass failed: {e}")
        return False

    # Check metrics
    print("\nChecking metrics...")
    try:
        metrics = wrapper.get_metrics()
        print("PASS: get_metrics() successful")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.6f}")
    except Exception as e:
        print(f"FAIL: get_metrics() failed: {e}")
        return False

    # Check bridge diagnostics
    print("\nChecking bridge diagnostics...")
    try:
        if hasattr(wrapper, 'bridge'):
            diag = wrapper.bridge.diagnostics()
            print("PASS: bridge.diagnostics() successful")
            for k, v in diag.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.6f}")
                elif isinstance(v, torch.Tensor):
                    print(f"   {k}: tensor on {v.device}")
                else:
                    print(f"   {k}: {v}")
    except Exception as e:
        print(f"FAIL: bridge.diagnostics() failed: {e}")
        return False

    # Final verification
    print("\n" + "=" * 60)
    print("ACTIVE MODE WORKING CORRECTLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_active_mode()
    sys.exit(0 if success else 1)
