"""HONESTY-1 regression tests.

The old cross_sample_null_zeta zero-pads control sinks whose sequence length
differs from the matched sink. Padding scales cosine toward zero (bounded by
sqrt(len_short / len_long)), which deflates the null distribution and
inflates gap / z_score. These tests (a) demonstrate that confound on
synthetic activations, (b) verify length_matched_null_zeta removes it, and
(c) pin the new function's honest behavior so it cannot silently regress.
"""

import math

import pytest
import torch

from unitarity_labs.core.metrics import (
    cross_sample_null_zeta,
    length_matched_null_zeta,
    manifold_coherence_zeta,
)

HIDDEN = 64


def _acts(seq_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq_len, HIDDEN, generator=gen)


class TestConfoundExists:
    """The old path really is length-confounded (documents the bug)."""

    def test_padding_caps_cosine_at_sqrt_length_ratio(self):
        # A control that IS the sink, truncated to half length, is a
        # maximally-similar control. Padding caps its score near
        # sqrt(len_ratio); length-matching recovers ~1.0.
        sink = _acts(32, seed=0)
        control = sink[:, :16, :].clone()

        padded_score = manifold_coherence_zeta(sink, control)
        # Exact identity: cos(x, pad(x[:L])) == ||x[:L]|| / ||x||, which
        # concentrates around sqrt(L / L_full) for iid activations. Either
        # way it is far below the true similarity of 1.0.
        norm_ratio = (
            control.flatten().norm() / sink.flatten().norm()
        ).item()
        assert padded_score == pytest.approx(norm_ratio, abs=1e-5), (
            f"padded cosine {padded_score:.4f} must equal the norm ratio "
            f"{norm_ratio:.4f}"
        )
        assert padded_score < 0.8, "padded score must be far below 1.0"

        res = length_matched_null_zeta(sink, sink, [control])
        assert res["null_mean"] > 0.999, (
            "length-matched score of a truncated copy must be ~1.0, got "
            f"{res['null_mean']:.4f}"
        )

    def test_old_null_std_deflated_versus_length_matched(self):
        # Random controls of varying (shorter) lengths: padding shrinks every
        # null score toward 0, so the old null_std is systematically smaller
        # than the honest length-matched null_std -> inflated z.
        source = _acts(48, seed=1)
        sink = _acts(48, seed=2)
        controls = [_acts(12 + 2 * i, seed=100 + i) for i in range(20)]

        with pytest.warns(UserWarning, match="HONESTY-1"):
            old = cross_sample_null_zeta(source, sink, controls)
        new = length_matched_null_zeta(source, sink, controls)

        assert old["null_std"] < new["null_std"], (
            "padding must deflate the null spread: old_std="
            f"{old['null_std']:.6f} new_std={new['null_std']:.6f}"
        )


class TestLengthMatchedBehavior:
    def test_equal_lengths_agree_with_old_path(self):
        # No length mismatch -> no confound -> both paths must agree.
        source = _acts(24, seed=3)
        sink = _acts(24, seed=4)
        controls = [_acts(24, seed=200 + i) for i in range(8)]

        old = cross_sample_null_zeta(source, sink, controls)
        new = length_matched_null_zeta(source, sink, controls)

        assert new["matched"] == pytest.approx(old["matched"], abs=1e-6)
        assert new["null_mean"] == pytest.approx(old["null_mean"], abs=1e-6)
        assert new["null_std"] == pytest.approx(old["null_std"], abs=1e-6)
        assert new["matched_len_tokens"] == 24
        assert new["length_matched"] is True

    def test_never_pads_matched_or_controls(self):
        # Mixed lengths: everything is truncated to the shortest (10 tokens);
        # matched score must equal the explicitly truncated computation.
        source = _acts(30, seed=5)
        sink = _acts(25, seed=6)
        controls = [_acts(10, seed=7), _acts(40, seed=8)]

        res = length_matched_null_zeta(source, sink, controls)
        assert res["matched_len_tokens"] == 10

        expected = manifold_coherence_zeta(
            source[:, :10, :], sink[:, :10, :]
        )
        assert res["matched"] == pytest.approx(expected, abs=1e-6)

    def test_rejects_mismatched_hidden_dim(self):
        source = _acts(16, seed=9)
        sink = _acts(16, seed=10)
        bad_control = torch.randn(1, 16, HIDDEN + 1)
        with pytest.raises(ValueError, match="hidden dimension"):
            length_matched_null_zeta(source, sink, [bad_control])

    def test_rejects_empty_controls(self):
        source = _acts(8, seed=11)
        sink = _acts(8, seed=12)
        with pytest.raises(ValueError, match="non-empty"):
            length_matched_null_zeta(source, sink, [])


class TestOldPathWarns:
    def test_warns_only_on_length_mismatch(self):
        source = _acts(16, seed=13)
        sink = _acts(16, seed=14)

        with pytest.warns(UserWarning, match="HONESTY-1"):
            cross_sample_null_zeta(source, sink, [_acts(12, seed=15)])

        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            cross_sample_null_zeta(source, sink, [_acts(16, seed=16)])
