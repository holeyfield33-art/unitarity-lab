"""Exact spectral references for short-context and cached-decode telemetry."""
import pytest
import torch
from unitarity_labs.core.bridge import CrossLayerEntanglementHook


def measured(x):
    bridge = object.__new__(CrossLayerEntanglementHook)
    bridge._source_activation = x
    bridge.lanczos_iter = 15
    return bridge.spectral_gap()


def reference(x):
    flat = x.double().reshape(-1, x.shape[-1])
    eigenvalues = torch.linalg.eigvalsh(flat.T @ flat / max(1, len(flat)))
    return float(eigenvalues[-1] - eigenvalues[-2]) if len(eigenvalues) > 1 else 0.


@pytest.mark.parametrize('shape', [(1, 32), (1, 768), (2, 32), (16, 32), (2, 4, 32), (64, 16)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_matches_exact_second_moment(shape, dtype):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(42)).to(dtype)
    assert measured(x) == pytest.approx(reference(x), rel=2e-4, abs=1e-5)


def test_repeated_rows_are_rank_one():
    x = torch.arange(1., 33.).repeat(8, 1)
    assert measured(x) == pytest.approx(float(x[0].square().sum()), rel=1e-5)


def test_repeated_largest_eigenvalues_have_zero_gap():
    assert measured(torch.eye(8)) == pytest.approx(0., abs=1e-6)


def test_zero_and_single_feature_preserve_existing_contract():
    assert measured(torch.zeros(1, 32)) == 0.
    assert measured(torch.ones(8, 1)) == 0.


def test_scale_squared_and_rng_unchanged():
    x = torch.randn(8, 32, generator=torch.Generator().manual_seed(8))
    state = torch.random.get_rng_state().clone()
    initial = measured(x)
    assert measured(3*x) == pytest.approx(9*initial, rel=1e-4)
    assert torch.equal(state, torch.random.get_rng_state())
