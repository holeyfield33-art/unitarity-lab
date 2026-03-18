"""
Geometric Brain Spectral Verification Protocol
Holeyfield-AI Collaboration — Aletheia Sovereign Systems, 2026

Tests whether transformer hidden states exhibit GUE spectral rigidity.
Pass condition: avg_r ≈ 0.578 ± 0.05, λ₂ > 0.1

Status: [COMPUTATIONAL] — requires live hidden states from universal_hook.py
"""

import pytest
import numpy as np
from scipy.linalg import eigvalsh


def compute_heat_kernel(h_states, sigma="auto"):
    """
    Constructs heat kernel W_ij from hidden states.
    Dynamic sigma maintains Cheeger constant > 0 (k=1 invariant bridge).
    """
    sq_dists = (
        np.sum(h_states**2, axis=1).reshape(-1, 1) +
        np.sum(h_states**2, axis=1) -
        2 * np.dot(h_states, h_states.T)
    )
    sq_dists = np.maximum(sq_dists, 0)  # numerical stability

    if sigma == "auto":
        # k=1 invariant: sigma = median nearest-neighbor distance
        # keeps manifold connected (Cheeger > 0)
        sigma = np.sqrt(np.median(np.sort(sq_dists, axis=1)[:, 1]))

    W = np.exp(-sq_dists / (2 * sigma ** 2))
    return W, sigma


def compute_r_ratio(h_states):
    """
    Computes normalized Laplacian eigenvalues and r-ratio spacing statistics.
    Returns avg_r, lambda_2, full eigenvalue spectrum.
    """
    W, sigma = compute_heat_kernel(h_states)

    # Symmetric normalized Laplacian — spectrum bounded [0, 2]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt

    # Sorted eigenvalues [0, λ₂, λ₃, ..., λ_n]
    evals = eigvalsh(L_sym)

    # r-ratio: min(δₙ, δₙ₊₁) / max(δₙ, δₙ₊₁)
    spacings = np.diff(evals)
    r_values = (
        np.minimum(spacings[1:], spacings[:-1]) /
        np.maximum(spacings[1:], spacings[:-1])
    )
    avg_r = float(np.mean(r_values))
    lambda_2 = float(evals[1])

    return avg_r, lambda_2, evals, sigma


@pytest.mark.geometric_brain
def test_spectral_rigidity_gue_attractor():
    """
    Verifies hidden state manifold exhibits GUE spectral rigidity.

    GUE target:     ⟨r⟩ ≈ 0.578 (GOE→GUE crossover)
    Poisson regime: ⟨r⟩ ≈ 0.386 (hallucination risk)
    Tolerance:      ±0.05 (finite-size N=512 effects)

    Requires: wrapper.get_buffer(layer=11) populated via
              register_geometric_hooks() before model forward pass.
    """
    pytest.importorskip("torch")
    from unitarity_labs.core.universal_hook import UniversalHookWrapper

    # Load buffer — must have run inference first
    try:
        wrapper = UniversalHookWrapper._instance
        h = wrapper.get_buffer(layer=11).squeeze(0).cpu().numpy()
    except (AttributeError, ValueError) as e:
        pytest.skip(f"No hook data available: {e}")

    assert h.shape[0] >= 512, (
        f"Sequence length {h.shape[0]} < 512. "
        "GUE convergence requires S ≥ 512 tokens (Marchenko-Pastur)."
    )

    avg_r, lambda_2, evals, sigma = compute_r_ratio(h)

    # Report
    print(f"\n{'='*50}")
    print(f"GEOMETRIC BRAIN AUDIT — LAYER 11")
    print(f"{'='*50}")
    print(f"⟨r⟩ measured:    {avg_r:.4f}")
    print(f"⟨r⟩ GUE target:  0.5996")
    print(f"⟨r⟩ crossover:   0.578")
    print(f"⟨r⟩ Poisson ref: 0.386")
    print(f"λ₂ connectivity: {lambda_2:.4f}")
    print(f"Auto sigma:      {sigma:.4f}")
    print(f"{'='*50}")

    if avg_r < 0.45:
        print("⚠️  POISSON DRIFT — context decoherence risk")
    elif abs(avg_r - 0.578) <= 0.05:
        print("✅ GUE ATTRACTOR — manifold rigid")
    else:
        print("⬜ GOE→GUE CROSSOVER — intermediate regime")

    # Assertions
    assert avg_r == pytest.approx(0.578, abs=0.05), (
        f"Manifold NOT rigid. ⟨r⟩={avg_r:.4f}, expected 0.578±0.05. "
        f"Drift toward Poisson={0.386} indicates context decoherence."
    )

    assert lambda_2 > 0.1, (
        f"Manifold shattering. λ₂={lambda_2:.4f} < 0.1. "
        "Algebraic connectivity too low for coherent context."
    )


@pytest.mark.geometric_brain
def test_poisson_detection():
    """
    Verifies the test correctly identifies Poisson-distributed states.
    Uses synthetic random (Poisson) hidden states as negative control.
    """
    np.random.seed(42)
    # Poisson-like: random uncorrelated hidden states
    h_random = np.random.randn(256, 64)

    avg_r, lambda_2, _, _ = compute_r_ratio(h_random)

    print(f"\nPoisson control ⟨r⟩: {avg_r:.4f} (expected ≈ 0.386)")

    # Random states should NOT pass GUE test
    assert abs(avg_r - 0.578) > 0.05, (
        f"Random states passed GUE test — test may be miscalibrated. "
        f"⟨r⟩={avg_r:.4f}"
    )


@pytest.mark.geometric_brain  
def test_heat_kernel_connectivity():
    """
    Verifies heat kernel produces a connected graph (Cheeger > 0).
    Dynamic sigma must keep λ₂ > 0.
    """
    np.random.seed(0)
    h = np.random.randn(128, 32)
    W, sigma = compute_heat_kernel(h, sigma="auto")

    assert sigma > 0, "Sigma must be positive"
    assert W.min() >= 0, "Heat kernel must be non-negative"
    assert np.allclose(W, W.T), "Heat kernel must be symmetric"

    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    evals = eigvalsh(L_sym)

    assert evals[1] > 1e-10, (
        f"Graph disconnected. λ₂={evals[1]:.2e}. "
        "Dynamic sigma failed to maintain connectivity."
    )
