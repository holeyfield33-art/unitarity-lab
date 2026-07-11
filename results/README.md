# results/

Evaluation outputs live under `results/runs/`.

## Layout

Each run gets its own directory:

```
results/runs/<date>_<sha>_<env>/
    gsm8k_real.json    # per-problem records + summary accuracy
    manifest.json      # git SHA, pip freeze, device name, seed, model, mode
```

- `<date>` — `YYYYMMDD`
- `<sha>`  — short git SHA of the commit the run was produced from
- `<env>`  — `cpu` or `cuda`

Produced by `python -m benchmarks.real_gsm8k` (see `benchmarks/real_gsm8k.py`).

## What's recorded (per problem)

`correct` (graded numeric match), `zeta_raw`, `cross_sample_null`
(`null_mean`/`null_std`/`gap`/`z_score`, controls = other problems' sink
activations), `spectral_gap`, measured `latency_ms`, `prompt_tokens`,
`completion_tokens`. No synthetic accuracy, no `time.sleep` latency.

## Version control

Run artifacts themselves (`results/runs/*/`) are **not** committed — they are
reproducible and `manifest.json` embeds a full `pip freeze`. Only this layout
description is tracked. Regenerate a run with the command above.
