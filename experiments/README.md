# Experiment desk: first measurements

Research snapshot: `66c004b5f37a5f4a129761ef1d7f87f306b100ac`.
InsideAI remains pinned to `1dfe4e52771eb2fd82f4def61d1e740f3c61b77d`.
These results describe this fresh clone, not a replacement of the running app's dependency.

## Follow-up: confirmed spectral-gap defect and experimental fix

The original findings and real-model numbers are preserved in
`metric_controls.baseline.json` and `layer_probe.baseline.json`. The `.results.json`
files now contain the rerun after the local `bridge.py` fix, including source-file
SHA-256 hashes to identify changes beyond the base Git revision.

Exact-reference tests initially failed **10 of 16 cases**. The Lanczos recurrence
does not maintain a fully orthogonal basis; on these low-rank inputs its largest
Ritz estimates were duplicated, making their difference nearly zero. The new path
uses the exact rank-one norm for one row, and the smaller Gram matrix when
`min(number_of_rows, hidden_dimension) <= 128`. Larger cases retain the existing
approximation and have not been made exact by this change.

The correction is local to `unitarity_labs/core/bridge.py`; it changes measurement,
not model tensors. The docstring now correctly identifies an uncentered second
moment and removes the unsupported interpretation that a small gap means weakening.

| Baseline prompt case | Original gap | Corrected gap |
|---|---:|---:|
| capital | 0.8125 | 638252.6875 |
| arithmetic | 0.625 | 838034.5 |
| grounded | 0.078125 | 167025.453125 |
| user transformer completion | 0.1875 | 638307.5 |

The 16 real-model pilot conditions were rerun. Their predicted top tokens and
target-token probabilities were unchanged. The synthetic one-row gap is now
37.84977, agreeing with the exact squared norm, rather than zero.

Validation: **126 passed** across `test_spectral_gap_reference.py`,
`test_bf16_paths.py`, and `test_criticality.py` (26 existing runtime warnings).
Reference tests cover float32/bfloat16, single/multiple/batched rows, repeated
rows, repeated eigenvalues, zero inputs, scale behavior, and RNG preservation.

Additional passive-hook integration tests could not collect: this checkout imports
`var_spectral`, but the available installed package uses `var`. The configured
package index returned no `var-spectral` distribution. A fresh sibling
`../VAR-experiments` clone at `31234551e524249a5e81453ec851c98ec8836fb7` also
contains `var`, not `var_spectral`. No import alias or silent substitution was used.
This packaging mismatch remains unresolved and prevents claiming a fully verified
new passive-hook deployment.

**Do not reuse previously calibrated spectral-gap thresholds with this fix.**
Before live integration, resolve the dependency mismatch, version the measurement,
reset/recalibrate the detector, and collect separate prefill/decode baselines.
For one-token inputs this uncentered spectral gap equals activation squared norm;
it is not an independent measurement of a rich spectral distribution.

### Actual KV-cache verification

`./experiments/run.ps1 decode` runs `decode_probe.py` on cached distilgpt2.
One prefill and three real cached decode steps passed against float64 exact
references. Maximum relative gap error was 2.03e-7. Decode tensors were
`[1, 1, 768]`, with gaps 7703.7441, 7054.8203, and 7129.4639. Cached predictions
also matched full-prefix recomputation within 6.39e-5 maximum log-probability
difference. Results and source hashes are in `decode_probe.results.json`.
This validates the corrected short-context path on actual model activations;
the remaining large-matrix approximation is still outside this verification.

### User's live Qwen GGUF integration

InsideAI now loads the user's local Qwen2.5-Coder 1.5B GGUF through Transformers.
The existing server reported 28 layers, 12 heads, dimension 1536 and CPU bfloat16.
Its loader edits were preserved; the corrected clone was not installed into it.
An observational pilot captured eight greedy grounded-answer trials, all exact
matches, including a red/blue context pair and an unknown-answer case. All repeated
outputs matched. All 32 unique telemetry steps were uncalibrated, so zero flags
provide no detector validation. The live old-metric means were prefill zeta
-0.0570875 and decode zeta 0.415671; these are measurements on a different
model/version and must not be pooled with the earlier distilgpt2 pilot.
See `../../Insideai/docs/EXPERIMENTS.md` for runnable capture commands and the
timestamped report path. Full events preserve 896 per-layer anomaly messages,
deduplicated into 32 shared step observations for analysis.

## Reproduce locally

From this clone in PowerShell:

```powershell
./experiments/run.ps1 metrics
./experiments/run.ps1 model
./experiments/run.ps1 tests
```

The runner uses the sibling Insideai Python environment for its existing scientific stack,
with this clone first on PYTHONPATH. Optional packages are isolated in `.experiment-deps`:
reedsolo 1.7.0, pyzmq 27.2.0, msgpack 1.2.2. To recreate that directory:

```powershell
../Insideai/backend/.venv/Scripts/python.exe -m pip install --target .experiment-deps reedsolo==1.7.0 pyzmq==27.2.0 msgpack==1.2.2
```

The model pilot uses cached distilgpt2 only, CPU float32, seed 42, two torch threads,
eval mode, eager attention, no KV cache. It does not connect to the running backend.
Model and repo revisions are in the result JSON. Each command overwrites its result JSON;
copy or commit results before changing conditions. The tests include socket-based local
dual-link checks; the two experiment scripts do not open network sockets.

## What the current screen means

- `L5`: zero-based sixth layer. `head μ`: mean attention across heads, not a particular head.
- Attention rows represent query positions; columns represent key positions. Attention
  alone does not demonstrate why an answer is correct or incorrect.
- `||h||`: L2 norm of the newest token's hidden vector. `Δresidual`: L2 distance from
  the preceding entry in Transformers' hidden-state tuple. These are unsigned magnitudes.
  In GPT-2 the final tuple entry includes final layer normalization, so the last UI delta
  is not a clean measurement of only the final block's residual update. The pilot instead
  captures each block's actual input/output and also records a relative delta.
- A sampled rank-15 token is possible under stochastic decoding. The most likely next
  token and the sampled token need not match. The displayed 14.5% belongs to the top token,
  not automatically to the selected token. Neither rank nor probability labels truth.
- `zeta_raw`: signed flattened source/sink cosine. Current wrapper selects source
  `num_layers // 2`, sink `max(0, num_layers - 2)` (L3/L4 for six layers).
- Spectral gap: estimated difference of the two largest absolute eigenvalues of
  **uncentered** `X.T @ X / n`. Despite the source docstring saying covariance, it does
  not subtract a mean. Lanczos approximates those eigenvalues.
- InsideAI's `flagged`: VAR's median/MAD rupture state on spectral gap. It is not BOCPD.
  The current stream repeats a shared step observation across layers.

## Synthetic mechanism probes

See `metric_controls.py` and `metric_controls.results.json`. These are constructed tensor
and signal inputs, not model hallucination labels.

| Probe | Observed result | Consequence for the experiment desk |
|---|---|---|
| Independent random tensors, then add shared offset 10 | zeta -0.0018 becomes 0.9900 | High raw cosine can reflect shared offsets; include matched-vs-shuffled controls. |
| Multiply source activation by 3 | spectral gap increases 9.0000 times | Log activation scale; evaluate trace-normalized gap alongside raw gap. |
| One-row source activation | production gap 0; exact rank-one gap is squared norm 37.8498 | Numerical/rank-handling discrepancy needs investigation before decode alarms are trusted. |
| Full-rank multi-token control | production gap 0.605375; exact 0.605378 | Failure is not a blanket failure on this full-rank control. |
| BOCPD synthetic change at index 150 | first alarm 150; zero pre-change alarms | Synthetic detector discrimination works on the chosen separated distributions. |
| Change zeta -1 to +1 | identical BOCPD probabilities | BOCPD currently uses r-ratio only; zeta argument is ignored. |
| Same sample after 20 stable vs 20 collapsed observations | both return 0.0460361 | Pre-truncation score lacks history dependence under current fixed likelihoods. |
| Chronos persistent +0.03 vs alternating +/-0.03 | persistent severs at index 6; alternating never severs | Signed drift cancels; use a separate absolute-jitter readout if jitter matters. |
| Dual sync on identical basis repeated six times | first three ~1.0, subsequent ~0.8 | Returned phi includes a policy penalty; expose raw cosine and adjusted control separately. |

BOCPD's pre-truncation recursion uses the same p0(x) for every growth hypothesis.
Summing posterior mass gives `P(reset) = h*p1(x) / ((1-h)*p0(x) + h*p1(x))`.
Thus the current sample controls the score independently of previous history in this regime.
Pruning can change this; the finding does not claim every possible configuration is identical.
The score should not be presented as a validated probability that an answer is false.

## Real model pilot

See `layer_probe.py` and `layer_probe.results.json`: four prompts, four conditions = 16
forward passes. Conditions are baseline, no-op sham, 90% L3 residual update, and zero L3
residual update. The intervention is `h_in + gain * (h_out - h_in)`, applied to L3's
block output. Handles are removed after every pass. Source/sink metrics are measured
after intervention. These are prefill measurements over the whole prompt.

| Target next token | Baseline probability | L3 attenuated | L3 blocked |
|---|---:|---:|---:|
| Paris | 0.2763% | 0.2468% | 0.0619% |
| 2 | 8.9884% | 8.8628% | 6.4049% |
| red | 0.2120% | 0.2205% | 0.2438% |

The user's transformer completion changed its top token from ` using` to ` the` when
L3 was blocked. No correctness target was assigned to this open-ended prompt.
All shams passed maximum log-probability difference < 1e-4 versus baseline.
Raw source/sink cosine stayed between approximately 0.979 and 0.997 across all conditions.

This pilot demonstrates that an intervention can alter predictions, with mixed target-token
effects. It does not measure answer accuracy: ` the` can precede a correct multi-token
answer, and distilgpt2 is being used as a completion model. No hallucination rate,
causal hallucination pathway, or effective blocking policy has been established.

## Verification

`tests/test_chronos_lock.py`, `tests/test_bocpd.py`, `tests/test_dual_link.py`:
**76 passed, 1 skipped**. The skipped test requires a real r-ratio collapse fixture.
Passing these tests does not resolve the measurement concerns above or validate distributed
LLM answer quality. The initial attempt lacked optional distributed dependencies; they
were then installed only into the clone's isolated dependency directory.

## Next experiments and UI integration

1. Short-context spectral reference checks and local fix are complete (above).
   Next validate actual KV-cached decode traces and the remaining large-matrix
   approximation. Track prefill/decode separately and version the corrected measurement.
2. Add a per-run manifest and event fields: model/repo revisions, node ID, prompt/seed,
   phase, layer/token, source/sink, calibration ID, intervention target/gain, timing,
   raw/adjusted sync, missing/stale-message counts. Show unavailable metrics explicitly.
3. Run a grounded QA set with known answers and unanswerable cases. Score full answers
   for support, correctness, and abstention. Split calibration and held-out prompts;
   compare signals against entropy, token probability, and simple activation norms.
4. Sweep layer, intervention gain, and seed with matched baseline and sham trials.
   For evidence of a pathway, use clean/corrupted prompt pairs and activation patching,
   then evaluate restoration of correct answers on held-out cases. Do not optimize zeta alone.
5. Chronos: vary peer lag, jitter, sequence jumps, and lost/corrupted messages. Measure
   detection delay and false severing independently of semantic answer scores.
6. Dual node: baseline A, baseline B, exchange-only, then bounded injection. Record actual
   exchange success, raw agreement, adjustment penalties, answer quality, and latency.
   A second agreeing node is not an independent truth label.
7. BOCPD: expose raw r-ratio and baseline/warmup metadata, audit history dependence,
   calibrate on real traces, then test held-out precision/recall and false interventions.
8. Add Experiment desk controls for explicit monitor-only, shadow intervention, and applied
   intervention modes. Start with reversible actions such as stop/abstain/retry; compare
   their costs and correctness before enabling automatic tensor intervention.

No deployed guardrail, running-server configuration, or installed telemetry pin was changed.
