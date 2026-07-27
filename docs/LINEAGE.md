# Lineage: what this project inherited, and what it did not

This note exists to settle a recurring question — *"what is the anchor, why is
every run a different number?"* — permanently, and to keep the project's public
claims tied to what its own data supports.

## The short version

`unitarity-lab` inherits a **method**, not a **constant**.

- **Methodological ancestor:** [`rh-solution`](https://github.com/holeyfield33-art/rh-solution)
  — a heuristic study of whether the Berry–Keating Hamiltonian needs `k = 1` to
  reproduce Riemann-zeta zero statistics.
- **What transferred:** GUE / spectral-statistics thinking, KS-testing,
  bootstrap null distributions, kill-test suites, committed manifests, and a
  disclaimers-first posture. That DNA is visible in VAR's W2 spectral detector
  and in this repo's audit harness.
- **What did *not* transfer:** `k = 1` itself, the Berry–Keating Hamiltonian, or
  any claim that transformer activations obey a number-theoretic invariant.
  `zeta_raw` here is cross-layer cosine coherence — the name is metaphor
  lineage only.

## Why `rh-solution`'s central result does not carry over

Per `rh-solution`'s own final release (v0.4-final), the `k = 1` heuristic **did
not survive its own kill-tests**: at N = 2000 zeros the GUE fit was rejected for
*all* k, the bootstrap showed the k-minimum was noise-dominated, and in a
sizeable fraction of sliding windows some `k ≠ 1` beat `k = 1`. It was closed,
correctly, as a **null result**. So there was never an empirically established
`k = 1`-style invariant to carry forward — and there is no analogue of one in
LLMs.

That is the honest answer to *"why is it a new number every time?"* **There is no
fixed invariant to anchor on, and there was never supposed to be.** Zeta zeros
are a fixed mathematical object; transformer activations are stochastic and
depend on prompt, seed, model, dtype, and hardware. Different numbers per run
are *expected*. The anchor in this repo is not a constant — it is:

1. the **committed manifest** (git SHA, environment, seed, effective dtype) that
   makes any single run reproducible, and
2. **distributions with stated error/false-positive rates**, not point values.

## Consistent with this repo's own audit

This lineage matches what `unitarity-lab`'s cross-model audit independently
found: the headline ζ result (z ≈ 13.5) is a **layer-adjacency artifact**, and
once depth is controlled the effect is statistically indistinguishable from a
length-matched null. See [`../results/audits/README.md`](../results/audits/README.md).

## The one rule this implies (do not break it)

Do **not** re-narrate the product as "built on Riemann-hypothesis math." Doing so
would attach an unverifiable claim to a repo whose own kill-tests refuted its
central heuristic — inverting the honesty that is the point of this suite.
`rh-solution` is **provenance, closed**: the origin of the *method*, not a
source of live claims or code to re-merge.
