# tests/fixtures — Real r-ratio trace baseline slot

This directory holds real r-ratio traces recorded from live model runs for
regression testing. The fixtures are intentionally excluded from git (add
`*.npy` to .gitignore if not already present) so the repo stays lightweight.

## Fixture format

| File | Description |
|------|-------------|
| `real_r_trace_stable.npy` | 1-D float64 array of r-ratios from a stable (GUE) run |
| `real_r_trace_collapse.npy` | 1-D float64 array of r-ratios from a run with an induced collapse |

Both files are produced by recording `r_ratio` values from
`Orchestrator.ingest()` StepRecords over a full session.

## How to generate

```python
import numpy as np
records = orchestrator._history
r_trace = np.array([r.r_ratio for r in records if r.r_ratio is not None])
np.save("tests/fixtures/real_r_trace_collapse.npy", r_trace)
```

## Enabling the real-data test

Once `real_r_trace_collapse.npy` is present, `test_real_baseline_detection`
in `tests/test_bocpd.py` will run automatically (the skipif guard checks for
the file). The test asserts:

1. The detector self-calibrates from the first 100 samples.
2. A changepoint with probability > 0.95 is detected somewhere after
   the stable-regime prefix.

Drop the file here and re-run `pytest tests/test_bocpd.py -v` to validate.
