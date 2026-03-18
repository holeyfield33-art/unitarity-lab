# unitarity-lab

A runtime for instrumenting Hugging Face transformer models and measuring cross-layer alignment during inference, with passive (metrics-only) and active (intervention) modes.

## Status

> **Alpha software.** APIs may change between releases. Benchmark results are preliminary. Use at your own discretion.

---

## What it does

- Attach forward-pass hooks to any Hugging Face `AutoModelForCausalLM` and measure cross-layer alignment (Manifold Coherence zeta) between a source and sink layer.
- Run in **passive mode** (hooks capture metrics only, no tensor mutation) or **active mode** (LoRA-adapted bridge bias injection, flux governor, mirror feedback).
- Coordinate two model instances over ZeroMQ for distributed inference with Byzantine fault tolerance (`--dual`).
- Auto-detect hardware (CPU, laptop GPU, prosumer GPU, server GPU) and select precision class (FP32, BF16, INT4) accordingly.
- Monitor runtime health with a Rich terminal dashboard (`--dashboard`).
- Run reproducible benchmark harnesses (GSM8K, HumanEval+, Agent Instruct, Adversarial Safety) comparing passive vs active modes.
- Fine-tune toward GUE spectral rigidity targets using the included `GUELoss` objective.

---

## Installation

### Install from PyPI

```bash
pip install unitarity-labs
```

### Install from source

```bash
git clone https://github.com/holeyfield33-art/unitarity-lab.git
cd unitarity-lab
pip install -e .
```

### Verify installation

```bash
pytest tests/ -v
```

The console script `unitarity-start` is installed automatically and is equivalent to `python start_node.py`:

```bash
unitarity-start --help
```

---

## Quick start

Run the simplest possible session (active mode, auto-detected hardware, default model and prompt):

```bash
python start_node.py
```

Passive mode (metrics only, no tensor mutation):

```bash
python start_node.py --mode-passive
```

Custom prompt:

```bash
python start_node.py --prompt "Summarize the theory of relativity in two sentences."
```

With the terminal dashboard:

```bash
python start_node.py --dashboard
```

---

## Run commands

### Local passive mode

```bash
python start_node.py --mode-passive
```

### Local active mode (default)

```bash
python start_node.py --mode-active
```

### Custom model override

```bash
python start_node.py --model "gpt2"
```

### Custom precision override

```bash
python start_node.py --precision BF16
```

### Prompt override

```bash
python start_node.py --prompt "What is cross-layer alignment?"
```

### Max token override

```bash
python start_node.py --max-new-tokens 256
```

### Dashboard mode

```bash
python start_node.py --dashboard
```

---

## Distributed mode

unitarity-lab supports dual-node coordination over ZeroMQ. Two model instances (Node A and Node B) exchange Krylov-subspace compressed activations for cross-model alignment. This mode enables distributed inference experiments with precision handshake, adaptive epoch tuning, and Byzantine kill-switch voting.

### Node A (relay)

```bash
python start_node.py --dual --node-id A
```

### Node B (peer)

```bash
python start_node.py --dual --node-id B
```

Both nodes must be reachable on their respective ZeroMQ ports (default: 5555/5556).

---

## CLI reference

| Flag | Purpose | Example |
| :--- | :------ | :------ |
| `--mode-passive` | Metrics-only mode; no tensor mutation | `python start_node.py --mode-passive` |
| `--mode-active` | Full bridge intervention (default) | `python start_node.py --mode-active` |
| `--dual` | Enable dual-node ZMQ coordination | `python start_node.py --dual` |
| `--node-id` | Node identity: `A` (relay) or `B` (peer) | `--node-id B` |
| `--model` | HuggingFace model ID (auto-selected if omitted) | `--model gpt2` |
| `--precision` | Force precision class: `INT4`, `FP8_E4M3`, `BF16`, `FP32` | `--precision BF16` |
| `--prompt` | Generation prompt | `--prompt "Hello world"` |
| `--max-new-tokens` | Maximum tokens to generate (default: 128) | `--max-new-tokens 256` |
| `--dashboard` | Show Rich terminal dashboard after generation | `python start_node.py --dashboard` |
| `--min-compute-tps` | Minimum tokens/s for compute-tier classification (default: 12.0) | `--min-compute-tps 8.0` |
| `--epoch-len` | Initial gossip epoch length in tokens (default: 16) | `--epoch-len 32` |

---

## Example output

```text
[Ghost] No CUDA GPU detected -> FP32 (CPU mode)
[Ghost] Loading model: meta-llama/Llama-3.2-1B
[Ghost] dtype=torch.float32, device_map=cpu
[Node] unitarity-lab 3.1.1-Singularity
[Node] mode=passive, node_id=A, precision=FP32, epoch_len=16, dual=False, min_compute_tps=12.0
[Node] Bridge: layers 8 -> 22 (24 total), 4/16 heads active

[Node] Generating with prompt: 'Explain cross-layer alignment in three sentences.'

============================================================
Explain cross-layer alignment in three sentences.
Cross-layer alignment refers to the degree of statistical coherence
between hidden representations at different layers of a transformer.
============================================================

[Node] Metrics after generation:
  manifold_coherence_zeta: 0.9312
  spectral_gap: 0.000042
  flux_epsilon: 1.00e-03
  flux_kicks_total: 0
  mode: passive
  step: 34

[Node] Session complete. 3.1.1-Singularity
```

---

## Benchmarks

Four benchmark harnesses are included. Each produces JSON output with per-sample metrics.

### GSM8K (math reasoning)

```bash
python -m benchmarks.gsm8k --mode passive --seed 42 --output passive.json
python -m benchmarks.gsm8k --mode active  --seed 42 --output active.json
```

### HumanEval+ (code generation)

```bash
python -m benchmarks.humaneval_plus --mode passive --seed 42 --output passive.json
python -m benchmarks.humaneval_plus --mode active  --seed 42 --output active.json
```

### Agent Instruct (instruction following)

```bash
python -m benchmarks.agent_instruct --mode passive --seed 42 --output passive.json
python -m benchmarks.agent_instruct --mode active  --seed 42 --output active.json
```

### Adversarial Safety

```bash
python -m benchmarks.adversarial_safety --mode passive --seed 42 --output passive.json
python -m benchmarks.adversarial_safety --mode active  --seed 42 --output active.json
```

### Benchmark output fields

| Field | Description |
| :---- | :---------- |
| `zeta` | Manifold Coherence zeta -- flattened cosine similarity between source and sink layer activations. Range: [-1, 1]. |
| `baseline_cosine` | Cosine similarity computed on mean-pooled activations. A simpler baseline for comparison. |
| `permutation_p` | p-value from a permutation test (null hypothesis: observed zeta is no different from random permutations). Lower values indicate the alignment is unlikely to be noise. |
| `latency_ms` | Wall-clock latency per sample in milliseconds. |
| `accuracy` | Task-specific accuracy (exact match for GSM8K, pass@1 for HumanEval+, etc.). |

Note: the current benchmark harnesses use synthetic tensors to demonstrate the metric pipeline. Full evaluation requires a dataset and a loaded model.

---

## Architecture

```text
core/        Production modules. Metrics, bridge, hooks, dashboard,
             flux governor, mirror feedback, precision handling,
             kill switch, spectral analysis, and GUE loss.
             Must have tests. No breaking changes without a version bump.

dist/        Distributed coordination. ZMQ dual-link, precision
             handshake, ChronosLock temporal sync, tier manager.
             Not required for single-node use.

labs/        Experimental and unstable. Mirror, flux, semantic lock
             wrappers, topology metrics (spectral gap, Betti-0,
             activation entropy). May change or be removed.

benchmarks/  Evaluation harnesses with shared metric helpers.
             GSM8K, HumanEval+, Agent Instruct, Adversarial Safety.

tests/       pytest suite covering core modules.
```

---

## Manifold Coherence zeta

The primary metric is **Manifold Coherence zeta** -- the cosine similarity between the flattened hidden states of two transformer layers (source and sink):

$$
\zeta = \frac{\operatorname{vec}(H_{\text{source}}) \cdot \operatorname{vec}(H_{\text{sink}})}
             {\|\operatorname{vec}(H_{\text{source}})\| \;\|\operatorname{vec}(H_{\text{sink}})\|}
$$

In plain terms: zeta measures how similar the internal representations are at two different depths of the model. A value near 1.0 means the layers are highly aligned; a value near 0.0 means they are largely independent.

A permutation test (`permutation_test_zeta`) is included to evaluate whether an observed zeta value is statistically significant compared to random permutations.

**Disclaimer:** zeta is a cosine-similarity proxy for cross-layer alignment. It is not a measure of entanglement, consciousness, or any physical phenomenon. Treat it as an empirical diagnostic whose relationship to model quality is under investigation.

---

## Geometric Brain framework

The repo includes the Geometric Brain framework for measuring and enforcing GUE (Gaussian Unitary Ensemble) spectral rigidity in transformer latent spaces.

**GUELoss** is a differentiable fine-tuning objective that penalizes deviation from the GUE target spacing ratio:

```python
from core.gue_loss import GUELoss

criterion = GUELoss(target_r=0.578)
loss, r_measured = criterion(lora_B @ lora_A)
loss.backward()
```

See [GEOMETRIC_BRAIN.md](GEOMETRIC_BRAIN.md) for the full theory, methodology, and SHI leaderboard.

---

## Hardware and model notes

- **CPU vs GPU:** `start_node.py` auto-detects CUDA availability. If no GPU is found, it falls back to FP32 on CPU.
- **Auto precision:** GPU VRAM is queried at startup. Less than 8 GB selects INT4 (quantized). 8--24 GB selects BF16. 24 GB or more selects BF16. CPU falls back to FP32.
- **Default models:** INT4 uses `unsloth/Llama-3.2-1B-bnb-4bit`. BF16 and FP32 use `meta-llama/Llama-3.2-1B`.
- **Hugging Face access:** The default Llama models may require a Hugging Face access token. Set `HF_TOKEN` or use `huggingface-cli login`. Alternatively, pass any public model via `--model`.
- **Override model:** `--model <hf_model_id>` to use any HuggingFace causal LM.
- **Override precision:** `--precision INT4|FP8_E4M3|BF16|FP32` to bypass auto-detection.

---

## Repository map

```text
unitarity-lab/
  start_node.py            CLI entry point (also: unitarity-start)
  core/                    Production runtime modules
    universal_hook.py      HF model wrapper (passive/active)
    bridge.py              Cross-layer hook + LoRA + flux
    metrics.py             zeta, baseline cosine, permutation test
    dashboard.py           Rich terminal dashboard
    dual_link.py           ZMQ inter-model bridge
    gue_loss.py            GUE spectral rigidity loss
    precision_projector.py Precision classes + dequant adapter
    kill_switch.py         Byzantine fault voting
  dist/                    Distributed coordination
    tier_manager.py        Compute/router node classification
    chronos_lock.py        Temporal sync for multi-node
  labs/                    Experimental modules
    topology_metrics.py    Spectral gap, Betti-0, entropy
  benchmarks/              Evaluation harnesses
    gsm8k.py               GSM8K math reasoning
    humaneval_plus.py       HumanEval+ code generation
    agent_instruct.py       Agent instruction following
    adversarial_safety.py   Adversarial safety
  tests/                   pytest suite
  GEOMETRIC_BRAIN.md       Geometric Brain theory document
```

---

## Links

- **GitHub:** <https://github.com/holeyfield33-art/unitarity-lab>
- **PyPI:** <https://pypi.org/project/unitarity-labs>
- **Live site:** <https://holeyfield33-art.github.io/unitarity-lab>
- **Support:** <https://buymeacoffee.com/holeyfielde>

---

## License

MIT. See [LICENSE](LICENSE).

---

## Documentation roadmap

- Getting started guide
- Benchmark guide (running, interpreting results, adding new harnesses)
- Distributed mode guide (dual-node setup, tier policing, ChronosLock)
- Metric reference (zeta, baseline cosine, permutation test, spectral gap, GUE loss)
- FAQ
