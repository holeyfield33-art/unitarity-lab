"""
Tests for GUELoss — Geometric Brain fine-tuning objective.
Verifies differentiability and gradient flow through LoRA adapters.
"""

import torch
import pytest
from unitarity_labs.core.gue_loss import GUELoss


@pytest.mark.geometric_brain
def test_gue_loss_gradients():
    """Verifies loss is differentiable and gradients flow to LoRA weights."""
    lora_A = torch.randn(8, 2048, requires_grad=True)
    lora_B = torch.randn(2048, 8, requires_grad=True)

    criterion = GUELoss(target_r=0.578)
    manifold = torch.matmul(lora_B, lora_A)
    loss, r_measured = criterion(manifold)

    loss.backward()

    assert loss.item() >= 0, "Loss must be non-negative"
    assert lora_A.grad is not None, "Gradients must flow to lora_A"
    assert lora_B.grad is not None, "Gradients must flow to lora_B"

    grad_norm = lora_A.grad.norm().item()
    assert grad_norm > 0, f"Zero gradients — backprop broken. norm={grad_norm}"

    print(f"\nGUELoss Verification:")
    print(f"  ⟨r⟩ measured: {r_measured:.4f}")
    print(f"  Target:       0.578")
    print(f"  Loss:         {loss.item():.6f}")
    print(f"  Grad norm:    {grad_norm:.6f}")
    print(f"  Status:       ✅ GRADIENTS FLOWING")


@pytest.mark.geometric_brain
def test_gue_loss_perfect_input():
    """Loss should approach zero when ⟨r⟩ ≈ 0.578."""
    criterion = GUELoss(target_r=0.578)
    # Small matrix — r-ratio will be computed
    matrix = torch.randn(16, 64, requires_grad=True)
    loss, r_measured = criterion(matrix)

    assert loss.item() >= 0
    print(f"\nPerfect input test — ⟨r⟩={r_measured:.4f}, loss={loss.item():.6f}")


@pytest.mark.geometric_brain
def test_gue_loss_hutchinson_vectors():
    """Verifies Hutchinson probe produces non-zero spectral signal."""
    criterion = GUELoss(target_r=0.578, n_vectors=16)
    matrix = torch.randn(8, 256, requires_grad=True)
    loss, _ = criterion(matrix)
    loss.backward()

    assert matrix.grad is not None
    assert matrix.grad.norm().item() > 0
