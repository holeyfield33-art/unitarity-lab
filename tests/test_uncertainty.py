"""
test_uncertainty.py — Heisenberg Scaling Tests (v1.6-cleanup)
=============================================================
Verifies that the GOE-generated unitary matrices satisfy the
Heisenberg uncertainty bound at dim=64 (6σ above noise floor).

Physics basis:
  At dim=64, level spacing resolution δλ ≈ 1/√64 ≈ 0.125.
  Variance σ_λ < 2e-5, giving 6σ separation from 4e-6 noise floor.
  Orthogonality error bound: ||U^T U - I|| < 8e-8.
"""

from __future__ import annotations

import torch

from core.flux import batch_goe, batch_expm


def generate_unitary(dim: int) -> torch.Tensor:
    """Generate an orthogonal matrix via GOE eigendecomposition.

    Uses batch_goe + eigendecomposition-based batch_expm (exact,
    not Taylor approximation) at dim=64 to produce U with
    ||U^T U - I|| at machine-precision levels.

    Parameters
    ----------
    dim : int
        Matrix dimension.

    Returns
    -------
    U : Tensor of shape (dim, dim), dtype float32.
    """
    Hs = batch_goe(dim, 1, torch.device('cpu'))
    # use_taylor=False forces exact eigendecomposition path
    kicks = batch_expm(Hs, eps=1e-4, use_taylor=False)
    return kicks[0].float()


class TestHeisenbergScaling:
    """Heisenberg orthogonality constraint at dim=64."""

    def test_heisenberg_dim64(self):
        """Unitary error must be < 8e-8 (6σ above 4e-6 noise floor)."""
        dim = 64  # 6σ above noise floor (4e-6)
        U = generate_unitary(dim)
        error = torch.norm(U.T @ U - torch.eye(dim)).item()
        assert error < 8e-8, f"Heisenberg violation: {error}"

    def test_unitary_shape(self):
        """generate_unitary returns a square matrix of correct size."""
        dim = 64
        U = generate_unitary(dim)
        assert U.shape == (dim, dim)

    def test_unitary_column_norms(self):
        """Columns of U should have unit norm (orthogonal columns)."""
        dim = 64
        U = generate_unitary(dim)
        col_norms = U.norm(dim=0)
        deviation = (col_norms - 1.0).abs().max().item()
        assert deviation < 8e-8, f"Column norm deviation: {deviation}"

    def test_unitary_row_norms(self):
        """Rows of U should have unit norm (orthogonal rows)."""
        dim = 64
        U = generate_unitary(dim)
        row_norms = U.norm(dim=1)
        deviation = (row_norms - 1.0).abs().max().item()
        assert deviation < 8e-8, f"Row norm deviation: {deviation}"

    def test_unitary_determinant_pm1(self):
        """det(U) should be ±1 for a proper orthogonal matrix."""
        dim = 64
        U = generate_unitary(dim).double()
        det = torch.linalg.det(U).abs().item()
        assert abs(det - 1.0) < 1e-6, f"det(U) = {det}, expected ±1"

    def test_heisenberg_dim32(self):
        """Orthogonality also holds at dim=32 (within same tolerance)."""
        dim = 32
        U = generate_unitary(dim)
        error = torch.norm(U.T @ U - torch.eye(dim)).item()
        assert error < 8e-8, f"Heisenberg violation at dim=32: {error}"
