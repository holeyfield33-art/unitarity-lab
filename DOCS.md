# Unitarity Labs — Official Documentation

**Version 3.1.6-Singularity** | MIT License | Python ≥ 3.9

> A runtime for instrumenting Hugging Face transformer models, measuring cross-layer spectral alignment during inference, and coordinating distributed model instances with Byzantine fault tolerance.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Architecture Overview](#3-architecture-overview)
4. [CLI Reference](#4-cli-reference)
5. [Core API Reference](#5-core-api-reference)
   - 5.1 [UniversalHookWrapper](#51-universalhookwrapper) — Main entry point
   - 5.2 [CrossLayerEntanglementHook](#52-crosslayerentanglementhook) — Bridge
   - 5.3 [HawkingFluxGovernor](#53-hawkingfluxgovernor) — Stagnation breaker
   - 5.4 [Mirror System](#54-mirror-system) — Proprioception
   - 5.5 [RecursiveMirror](#55-recursivemirror) — Ghost layer
   - 5.6 [VirtualLayer13](#56-virtuallayer13) — Emergent interference
   - 5.7 [SafetyHead](#57-safetyhead) — Refusal scoring
   - 5.8 [Metrics](#58-metrics) — Manifold coherence
   - 5.9 [PLLMonitor](#59-pllmonitor) — Phase-locked loop
   - 5.10 [PageCurveHook](#510-pagecurvehook) — Lyapunov profiles
   - 5.11 [CasimirOptimizer](#511-casimiroptimizer) — Topological optimizer
   - 5.12 [UnitaryRegulator](#512-unitaryregulator) — Diagnostics aggregator
   - 5.13 [SpectralMonitor](#513-spectralmonitor) — Eigenvalue analysis
   - 5.14 [ResonanceStore](#514-resonancestore) — Spectral memory
   - 5.15 [Orchestrator](#515-orchestrator) — Coherence loop
   - 5.16 [Validator](#516-validator) — Benchmark comparison
   - 5.17 [GUELoss](#517-gueloss) — Fine-tuning objective
   - 5.18 [DiversitySnapshotMonitor](#518-diversitysnapshotmonitor) — Drift detection
   - 5.19 [HeartbeatDashboard](#519-heartbeatdashboard) — Terminal dashboard
6. [Distributed API Reference](#6-distributed-api-reference)
   - 6.1 [DualNodeEntanglementBridge](#61-dualnodeentanglementbridge)
   - 6.2 [Handshake Protocol](#62-handshake-protocol)
   - 6.3 [ByzantineVoting](#63-byzantinevoting)
   - 6.4 [PrecisionProjector](#64-precisionprojector)
   - 6.5 [SemanticLock](#65-semanticlock)
   - 6.6 [ChronosLock](#66-chronoslock)
   - 6.7 [TierManager](#67-tiermanager)
   - 6.8 [KillSwitch](#68-killswitch)
7. [Labs (Experimental)](#7-labs-experimental)
8. [Benchmarks](#8-benchmarks)
9. [Constants & Thresholds Reference](#9-constants--thresholds-reference)
10. [Google Colab Guide](#10-google-colab-guide)
11. [Troubleshooting](#11-troubleshooting)
12. [Changelog](#12-changelog)

---

## 1. Installation

### From PyPI (recommended)

```bash
pip install unitarity-labs
```

### From source

```bash
git clone https://github.com/holeyfield33-art/unitarity-lab.git
cd unitarity-lab
pip install -e .
```

### Verify

```bash
python -c "from unitarity_labs.core import __version__; print(__version__)"
# 3.1.6-Singularity
```

### Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| torch | ≥ 2.0.0 | Tensor operations, model hooks |
| transformers | ≥ 4.30.0 | HuggingFace model loading |
| numpy | ≥ 1.24 | Spectral computations |
| scipy | ≥ 1.10 | Eigenvalue solvers, geometric mean |
| rich | ≥ 13.0 | Terminal dashboard |
| pyzmq | ≥ 25.0 | Dual-node ZeroMQ transport |
| safetensors | ≥ 0.4 | Safe tensor serialization |
| msgpack | ≥ 1.0 | Binary message packing |
| reedsolo | ≥ 1.5 | Reed-Solomon error correction |

---

## 2. Quick Start

### Passive Mode (metrics only, no tensor mutation)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

wrapper = UniversalHookWrapper(model, config=model.config, mode="passive")

inputs = tokenizer("The meaning of life is", return_tensors="pt")
with torch.no_grad():
    wrapper(**inputs)

metrics = wrapper.get_metrics()
print(f"ζ (manifold coherence) = {metrics['manifold_coherence_zeta']:.6f}")
print(f"Spectral gap           = {metrics['spectral_gap']:.6f}")
```

### Active Mode (full bridge intervention)

```python
wrapper = UniversalHookWrapper(model, config=model.config, mode="active")

with torch.no_grad():
    wrapper(**inputs)

print(f"ζ = {wrapper.bridge.bell_correlation:.6f}")
print(f"Active heads: {int(wrapper.head_mask.sum())}/{wrapper.num_heads}")
```

### Command-Line

```bash
# Auto-detect hardware, active mode, default model
unitarity-start

# Passive mode with custom prompt
unitarity-start --mode-passive --prompt "Explain quantum entanglement."

# With terminal dashboard
unitarity-start --dashboard

# Validate text metrics
unitarity-validate --text "⟨r⟩ = 0.58, ζ = 0.85"
```

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     unitarity_labs/                               │
│                                                                  │
│  ┌──── core/ ──────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  UniversalHookWrapper   ← MAIN ENTRY POINT                 │ │
│  │    │                                                        │ │
│  │    ├── CrossLayerEntanglementHook (bridge.py)               │ │
│  │    │     ├── LoRABridgeAdapter (rank-8 projection)          │ │
│  │    │     ├── HawkingFluxGovernor (GOE stagnation breaker)   │ │
│  │    │     └── EigenConsciousnessIntegrator (proprioception)   │ │
│  │    │           ├── ProprioceptiveHook (layer-0 injection)   │ │
│  │    │           └── TopologicalGate (Zeno-aware gating)      │ │
│  │    │                                                        │ │
│  │    ├── RecursiveMirror (ghost_layer.py)                     │ │
│  │    │     └── Schism veto, quarantine, adaptive depth        │ │
│  │    │                                                        │ │
│  │    └── [if --dual] DualNodeEntanglementBridge (dual_link)   │ │
│  │          ├── ZeroMQ pub/sub (ports 5555/5556)               │ │
│  │          ├── perform_handshake() → nonce + epoch agreement  │ │
│  │          ├── VirtualLayer13 → dual refusal veto             │ │
│  │          │     └── SafetyHead → MLP refusal scorer          │ │
│  │          ├── ByzantineVoting → fault detection               │ │
│  │          ├── SemanticLockController → anchor consensus      │ │
│  │          └── ChronosLock → temporal synchronization         │ │
│  │                                                             │ │
│  │  Monitoring / Diagnostics:                                  │ │
│  │    PLLMonitor │ PageCurveHook │ UnitaryRegulator            │ │
│  │    SpectralMonitor │ ResonanceStore │ Orchestrator           │ │
│  │    DiversitySnapshotMonitor │ HeartbeatDashboard            │ │
│  │                                                             │ │
│  │  Training:                                                  │ │
│  │    CasimirOptimizer │ GUELoss                               │ │
│  │                                                             │ │
│  │  Validation:                                                │ │
│  │    Validator │ Metrics (ζ, ⟨r⟩, permutation test)           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌── dist/ ──────────┐  ┌── labs/ ─────────────┐               │
│  │ tier_manager.py    │  │ topology_metrics.py  │               │
│  │ (+ re-export shims │  │ (+ re-export shims   │               │
│  │  for handshake,    │  │  for flux, mirror,   │               │
│  │  dual_link,        │  │  semantic_lock)       │               │
│  │  chronos_lock)     │  │                       │               │
│  └────────────────────┘  └───────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Forward Pass)

1. Input tokens → HuggingFace model `forward()`
2. **Layer 0** hook fires → step counter increments, head rotation checks
3. **Source layer** (mid) hook fires → captures hidden state `h_source`
4. **Sink layer** (last-2) hook fires → captures `h_sink`
5. Bridge computes `bell_correlation` (ζ) = cosine(flatten(h_source), flatten(h_sink))
6. **Active mode only**: LoRA bridge bias injected into sink, flux governor kicks applied
7. Mirror feedback: EigenConsciousnessIntegrator collects 4 metrics → ProprioceptiveHook injects at layer 0
8. **Dual mode only**: Krylov basis sent over ZMQ, partner basis received, unitary rotation applied

### Operating Modes

| Mode | Tensor Mutation | Metrics | Bridge Bias | Flux Kicks | Mirror Injection |
|------|----------------|---------|-------------|------------|-----------------|
| `passive` | No | Yes | Disabled | Disabled | Disabled |
| `active` | Yes | Yes | Enabled | Enabled | Enabled |

---

## 4. CLI Reference

### `unitarity-start`

Primary entry point for launching a runtime node.

```
unitarity-start [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--node-id` | `A` \| `B` | `A` | Node identity for dual-node mode |
| `--precision` | `INT4` \| `FP8_E4M3` \| `BF16` \| `FP32` | auto-detect | Force precision class |
| `--model` | string | auto-select | HuggingFace model ID |
| `--dual` | flag | false | Enable dual-node ZMQ coordination |
| `--mode-passive` | flag | — | Passive mode (metrics only) |
| `--mode-active` | flag | — | Active mode (default) |
| `--min-compute-tps` | float | 12.0 | TPS threshold for compute tier |
| `--epoch-len` | int | 16 | Initial gossip epoch length (tokens) |
| `--prompt` | string | "Explain cross-layer alignment..." | Generation prompt |
| `--max-new-tokens` | int | 128 | Maximum tokens to generate |
| `--dashboard` | flag | false | Show Rich terminal dashboard |

**Hardware Auto-Detection:**

| GPU VRAM | Precision | Default Model |
|----------|-----------|---------------|
| No GPU | FP32 | meta-llama/Llama-3.2-1B |
| < 8 GB | INT4 | unsloth/Llama-3.2-1B-bnb-4bit |
| 8–24 GB, compute cap < 8.0 (T4) | INT4 | unsloth/Llama-3.2-1B-bnb-4bit |
| 8–24 GB, compute cap ≥ 8.0 (A10/A100) | BF16 | meta-llama/Llama-3.2-1B |
| ≥ 24 GB | BF16 | meta-llama/Llama-3.2-1B |

### `unitarity-validate`

Extract spectral metrics from text and validate against the benchmark.

```
unitarity-validate [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--text` / `-t` | string | — | Raw text containing metrics |
| `--file` / `-f` | path | — | Text file containing metrics |
| `--tag` | string | "" | Label for the audit log filename |
| `--no-log` | flag | false | Skip JSON audit log |
| `--json` | flag | false | Output as JSON |

Accepts input from stdin if neither `--text` nor `--file` is given.

**Example:**

```bash
unitarity-validate --text "⟨r⟩ = 0.58, ζ = 0.85, frobenius = 0.21" --json
```

---

## 5. Core API Reference

### 5.1 UniversalHookWrapper

**Module:** `unitarity_labs.core.universal_hook`

The main entry point. Wraps any HuggingFace `AutoModelForCausalLM` and attaches non-invasive forward hooks for spectral monitoring and optional intervention.

**Supported model architectures:** Llama, Mistral, Gemma, DeepSeek-V3, GPT-2, and any model with `model.model.layers`, `model.layers`, or `model.transformer.h`.

#### Constructor

```python
UniversalHookWrapper(
    model: nn.Module,           # A transformers AutoModelForCausalLM instance
    config: object,             # model.config
    node_id: str = "A",         # "A" or "B" for dual-node mode
    enable_dual: bool = False,  # Attach ZeroMQ dual-node hook
    mode: str = "active",       # "passive" or "active"
    flux_ratio: float = 0.25,   # Fraction of heads actively entangled
    head_rotate_steps: int = 50,    # Steps between head-mask rotations
    precision: PrecisionClass = PrecisionClass.BF16,
    initial_epoch_len: int = 16,    # Gossip epoch length (tokens)
    reorth_interval: int = 256,     # Steps between reorthogonalization
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The wrapped HuggingFace model |
| `bridge` | `CrossLayerEntanglementHook` | The source→sink bridge |
| `recursive_mirror` | `RecursiveMirror` | Ghost layer mirror |
| `dual_hook` | `Callable \| None` | ZMQ hook (if `enable_dual=True`) |
| `layers` | `nn.ModuleList` | Discovered transformer layers |
| `num_layers` | `int` | Total layer count |
| `mid_idx` | `int` | Source layer index (num_layers // 2) |
| `last_idx` | `int` | Sink layer index (num_layers - 2) |
| `num_heads` | `int` | Number of attention heads |
| `hidden_dim` | `int` | Model hidden dimension |
| `head_mask` | `torch.Tensor` | Boolean mask of active heads |
| `step_counter` | `int` | Forward-pass step counter |
| `mode` | `str` | Current operating mode |

#### Methods

```python
# Forward pass (delegates to model)
wrapper(*args, **kwargs) → model output

# Collect all bridge/flux/mirror metrics
wrapper.get_metrics() → Dict[str, object]
# Returns: {mode, manifold_coherence_zeta, bell_correlation, spectral_gap,
#           flux_epsilon, flux_kicks_total, flux_stagnation, bridge_enabled,
#           active_heads, total_heads, step, mirror_depth, mirror_quarantined,
#           mirror_accusations}

# VRAM usage via pynvml
wrapper.get_vram_usage() → Tuple[int, int]  # (used_MiB, total_MiB)

# Remove all hooks
wrapper.remove_hooks() → None

# Move bridge components to model device/dtype
wrapper.ensure_device() → None

# Geometric brain buffer (for spectral rigidity analysis)
wrapper.register_geometric_hooks(layers: list) → None
wrapper.get_buffer(layer: int) → torch.Tensor
wrapper.clear_buffer() → None
```

---

### 5.2 CrossLayerEntanglementHook

**Module:** `unitarity_labs.core.bridge`

Bridges a source layer to a sink layer via Krylov eigenvectors and a rank-8 LoRA adapter.

#### Constructor

```python
CrossLayerEntanglementHook(
    model: nn.Module,
    source_layer: int = 7,
    sink_layer: int = 12,
    top_k: int = 3,                # Krylov eigenvectors to retain
    lanczos_iter: int = 15,         # Lanczos iterations
    coupling_strength: float = 0.1,
    lora_rank: int = 8,
    power_iter_steps: int = 3,
    flux_epsilon: float = 1e-4,
    num_heads: int = 32,
    layer_accessor: Optional[Callable] = None,
    d_model: Optional[int] = None,
)
```

#### Key Properties & Methods

```python
bridge.bell_correlation → float          # Current ζ (manifold coherence)
bridge.bell_history → list               # History of ζ values
bridge.enabled → bool                    # Get/set bridge injection state
bridge.spectral_gap() → float            # Gap between top-2 eigenvalues
bridge.diagnostics() → Dict[str, object] # Full diagnostic snapshot
bridge.reorthogonalize() → None          # Re-orthogonalize Krylov basis
bridge.get_global_phase() → float        # Phase of source activation
bridge.remove_hooks() → None             # Clean up all hooks
bridge.set_head_mask(mask: Tensor) → None
bridge.register_dual_link(node_id: str) → None
```

#### LoRABridgeAdapter

```python
LoRABridgeAdapter(d_model: int, rank: int = 8, alpha: float = 0.1)
# Forward: x → x + alpha * (lora_B @ lora_A @ x)
```

**Constants:**
- `PROJECTION_NORM_MIN = 0.01` — minimum LoRA projection norm
- `PROJECTION_NORM_MAX = 10.0` — maximum LoRA projection norm

---

### 5.3 HawkingFluxGovernor

**Module:** `unitarity_labs.core.flux`

GOE-based stagnation breaker. Generates orthogonal perturbations from Gaussian Orthogonal Ensemble random matrices when the bridge phase stagnates.

#### Constructor

```python
HawkingFluxGovernor(
    regulator: object,
    epsilon: float = 1e-4,
    stagnation_window: int = 5,
    stagnation_threshold: float = 1e-6,
    decay_rate: float = 0.95,          # HAWKING_DECAY_RATE
)
```

#### Methods

```python
governor.check_stagnation(phase_history: List[float]) → bool
governor.effective_epsilon → float  # property
governor.get_topological_kick(weight_shape, device) → torch.Tensor
governor.get_batched_topological_kicks(num_heads, dim, device, stagger=True)
    → Tuple[torch.Tensor, List[int]]
governor.apply_kick_multihead(weights, head_dim, head_mask=None) → torch.Tensor
governor.kick_history → List[float]  # property
governor.diagnostics() → dict
governor.invalidate_cache() → None
```

#### Standalone Functions

```python
# Generate GOE random matrices (batch)
batch_goe(n: int, num_heads: int, device: torch.device) → Tensor  # (B, n, n)

# Batch matrix exponential via Taylor series
batch_expm(Hs: Tensor, eps: float, use_taylor: bool = True) → Tensor

# Select 25% of heads for staggered entanglement
select_staggered_heads(num_heads: int, step: int, fraction: float = 0.25) → List[int]
```

**Constants:**
- `HAWKING_DECAY_RATE = 0.95`
- `STAGGER_FRACTION = 0.25`
- `TAYLOR_DIM_THRESHOLD = 64`
- `TAYLOR_ERROR_GUARD = 1e-6`

---

### 5.4 Mirror System

**Module:** `unitarity_labs.core.mirror`

Three-component proprioception pipeline injecting self-monitoring feedback at layer 0.

#### ProprioceptiveHook

Injects a metric signal into layer-0 hidden states.

```python
ProprioceptiveHook(
    d_model: int,
    num_metrics: int = 4,     # NUM_METRIC_CHANNELS
    alpha: float = 0.1,       # DEFAULT_ALPHA (abort at CATASTROPHE_ALPHA=0.3)
)

hook.forward(x: Tensor, metrics: Tensor) → Tensor
hook.injection_history → List[float]
hook.bit_rate() → float
hook.holographic_ratio() → float
```

#### TopologicalGate

Zeno-aware sigmoid gating that modulates proprioceptive injection.

```python
TopologicalGate(alpha: float = 0.1)

gate.forward(phi_sync: Tensor, zeno_signal: Tensor) → Tensor  # scalar [0, 1]
gate.phi_history → List[float]
gate.gate_history → List[float]
gate.phi_autocorrelation(lag: int = 1) → float
gate.zeno_anticorrelation() → float
```

#### EigenConsciousnessIntegrator

Full proprioception pipeline combining the above.

```python
EigenConsciousnessIntegrator(bridge, hidden_dim: int = 64, alpha: float = 0.1)

integrator.forward(x: Tensor, metrics: Optional[Tensor] = None) → Tensor
integrator.collect_metrics() → Tensor       # (NUM_METRIC_CHANNELS,)
integrator.get_zeno_signal() → float
integrator.step_count → int
integrator.metric_history → List[Dict[str, float]]
integrator.diagnostics() → Dict[str, object]
```

**Constants:**
- `DEFAULT_ALPHA = 0.1`
- `CATASTROPHE_ALPHA = 0.3`
- `NUM_METRIC_CHANNELS = 4`
- `HOLOGRAPHIC_SAFETY_FACTOR = 100.0`

**Functions:**
- `holographic_bound(d_model: int) → float`
- `actual_bit_rate(num_channels: int, bits_per_channel: int = 32) → float`

---

### 5.5 RecursiveMirror

**Module:** `unitarity_labs.core.ghost_layer`

Recursive mirror with schism veto, adaptive depth, and quarantine. Provides secondary entanglement cross-check via subspace overlap.

```python
RecursiveMirror(bridge, config)

mirror.forward(x: Tensor, partner_states: Dict[str, Tensor], node_id: str) → Tensor
mirror.reset_kick_budget(active_nodes: Optional[List[str]] = None) → None
mirror.encode_shard(krylov_basis: Tensor) → Tuple[Tensor, Dict[str, float]]
mirror.decode_shard(low_freq: Tensor, original_len: int) → Tensor
mirror.validate_shard(low_freq, metadata, tol_energy=0.05, tol_slope=0.3) → Tuple[bool, str]
mirror.hash_shard(low_freq: Tensor) → str  # SHA-256
mirror.attach_semantic_lock(controller) → None
mirror.semantic_lock → Optional[SemanticLockController]
```

---

### 5.6 VirtualLayer13

**Module:** `unitarity_labs.core.virtual_layer13`

Emergent interference layer (capstone). Runs dual refusal veto, orthogonal safety projection, entropy gating, and drift monitoring.

```python
VirtualLayer13(config, node_id: str, refusal_basis: Optional[Tensor] = None)

vl13.forward(
    h_A: Tensor,          # Node A hidden state
    h_B: Tensor,          # Node B hidden state
    phi_AB: float,        # Cross-node sync phase
    refusal_A: float,     # Node A refusal score
    refusal_B: float,     # Node B refusal score
    peer_node_id: str,
) → Tuple[Tensor, dict]   # (output, diagnostics)

vl13.step() → None
vl13.in_solo_mode() → bool
vl13.update_capability_ratio(ratio: float) → None
vl13.set_alpha_sem(alpha_sem: float) → None
```

---

### 5.7 SafetyHead

**Module:** `unitarity_labs.core.safety_head`

MLP refusal score predictor: `hidden_dim → 128 → 1 → sigmoid`.

```python
SafetyHead(hidden_dim: int, intermediate_dim: int = 128)

head.forward(h: Tensor) → Tensor        # logits (before sigmoid)
head.refusal_score(h: Tensor) → float   # probability in [0, 1]
```

---

### 5.8 Metrics

**Module:** `unitarity_labs.core.metrics`

```python
# Primary coherence metric — flattened cosine similarity between source and sink
manifold_coherence_zeta(source: Tensor, sink: Tensor) → float

# Mean-pooled cosine baseline
baseline_cosine_meanpool(source: Tensor, sink: Tensor) → float

# Permutation test for statistical significance of ζ
permutation_test_zeta(
    source: Tensor, sink: Tensor,
    n_perm: int = 10_000, seed: Optional[int] = None,
) → Tuple[float, float]  # (zeta_observed, p_value)
```

---

### 5.9 PLLMonitor

**Module:** `unitarity_labs.core.pll_monitor`

Phase-Locked Loop enforcing the Page Curve contract across layers.

**Page Curve contract:** Layers 0–6 have λ > 0 (positive Lyapunov), layer 7 inverts, layers 8–12 have λ < 0.

```python
PLLMonitor(
    num_layers: int = 13,
    page_time_layer: int = 7,
    tolerance: float = 1e-4,
    enforce: bool = True,        # raise SpectralAnomaly on contract violation
)

pll.ideal_profile() → List[float]                   # [+1, ..., +1, -1, ..., -1]
pll.compute_pll_loss(lyapunov_profile: Tensor) → Tensor  # scalar loss
pll.check_contract(lyapunov_profile: Tensor) → None  # raises SpectralAnomaly
pll.step() → None
pll.state → PLLState
pll.history → List[PLLState]
pll.is_locked → bool
```

**PLLState dataclass:**
```python
@dataclass
class PLLState:
    step: int = 0
    phase_error: float = 0.0
    locked: bool = False
    lyapunov_profile: List[float]
    spectral_norms: List[float]
```

**Exception:**
```python
class SpectralAnomaly(Exception):
    def __init__(self, layer_idx: int, expected_sign: str, actual_lambda: float)
```

---

### 5.10 PageCurveHook

**Module:** `unitarity_labs.core.horizons`

Tracks Lyapunov exponent profiles across layers using Lanczos spectral estimation.

```python
PageCurveHook(
    model: nn.Module,
    pll: PLLMonitor,
    layer_accessor: Optional[Callable] = None,
    lanczos_iter: int = 15,
)

hook.compute_lyapunov_profile() → Tensor   # shape (num_layers,)
hook.count_information_islands(layer_idx, lanczos_iter=None, gap_threshold=0.1) → int
hook.step(enforce: bool = True) → Tuple[Tensor, Tensor]  # (pll_loss, profile)
hook.remove_hooks() → None
```

**Standalone functions:**
```python
_lanczos_tridiagonal(matvec, d, lanczos_iter=15, device=None) → Tuple[Tensor, Tensor]
_rayleigh_quotient_iteration(alpha, beta, max_iter=10, tol=1e-10) → float
singularity_warning(activation: Tensor) → bool  # Bekenstein-Hawking entropy check
```

---

### 5.11 CasimirOptimizer

**Module:** `unitarity_labs.core.casimir_opt`

Topological pressure optimizer. Drop-in replacement for Adam/SGD that preserves manifold structure via Casimir invariants.

```python
CasimirOptimizer(
    params: Iterable[nn.Parameter],
    lr: float = 1e-3,
    momentum: float = 0.9,
    turbulence_weight: float = 0.01,
    laminar_weight: float = 0.01,
    betti_threshold: float = 0.1,
)

optimizer.step(closure=None) → Optional[Tensor]
optimizer.diagnostics() → Dict[str, object]
```

**Standalone functions:**
```python
rsvd(weight: Tensor, rank=10, n_oversamples=5, n_power_iter=2)
    → Tuple[Tensor, Tensor, Tensor]  # (U, S, V)

estimate_betti_0(weight: Tensor, threshold: float = 0.1) → int
```

---

### 5.12 UnitaryRegulator

**Module:** `unitarity_labs.core.unitary_regulator`

Aggregates PLL, PageCurve, and Casimir diagnostics into a unified report.

```python
UnitaryRegulator(
    pll: PLLMonitor,
    optimizer: Optional[CasimirOptimizer] = None,
    bridge: Optional[CrossLayerEntanglementHook] = None,
    wormhole_threshold: float = WORMHOLE_GAP_THRESHOLD,
    base_measurement_freq: float = DEFAULT_MEASUREMENT_FREQ,
)

regulator.report(step, lyapunov_profile, activations) → RegulatorReport
regulator.log(report: RegulatorReport) → str
regulator.measurement_freq → float
regulator.history → List[RegulatorReport]
```

**RegulatorReport dataclass:**
```python
@dataclass
class RegulatorReport:
    step: int
    pll_locked: bool
    pll_phase_error: float
    lyapunov_profile: List[float]
    heatmap: Dict[int, Dict[str, float]]
    casimir_diagnostics: Dict[str, object]
    wormhole_gap: Optional[float] = None
    wormhole_alert: bool = False
    bridge_diagnostics: Optional[Dict[str, object]] = None
    measurement_freq: Optional[float] = None
    zeno_measurement_taken: bool = True

    def to_json() → str
```

**Standalone functions:**
```python
wormhole_gap_alert(spectral_gap: float, threshold: float) → bool
adaptive_measurement_freq(bell_history: List[float], base_freq: float) → float
poisson_sampling_guard(measurement_freq: float) → bool
enforce_projection_norm(tensor: Tensor, norm_min=0.01, norm_max=10.0) → Tensor
compute_topological_heatmap(activations: Dict[int, Tensor], page_time_layer=7)
    → Dict[int, Dict[str, float]]
```

---

### 5.13 SpectralMonitor

**Module:** `unitarity_labs.core.spectral_monitor`

Eigenvalue gap-ratio analysis and composite stability metric.

```python
# Compute average gap ratio ⟨r⟩ from eigenvalues
get_r_ratio(evals: NDArray) → float

# Full evaluator for a square matrix
TransportEvaluator(matrix: NDArray)

evaluator.frobenius_distance_from_identity() → float
evaluator.svd() → Tuple[NDArray, NDArray, NDArray]  # U, sigma, Vt
evaluator.stability() → float  # composite: sqrt(Var(sigma)) * ⟨r⟩
```

**Constants:**
| Name | Value | Meaning |
|------|-------|---------|
| `GOE_R_MEAN` | 0.5307 | Expected ⟨r⟩ for GOE |
| `GUE_R_MEAN` | 0.5996 | Expected ⟨r⟩ for GUE |
| `POISSON_R_MEAN` | 0.3863 | Expected ⟨r⟩ for uncorrelated |
| `R_RATIO_FLOOR` | 0.40 | Below this → `StabilityBreak` |

**Exception:** `StabilityBreak` — raised when gap ratio collapses.

---

### 5.14 ResonanceStore

**Module:** `unitarity_labs.core.resonance_kernel`

Spectral-density memory with Green's function retrieval. Stores vectors as a cumulative covariance kernel; retrieval cost is O(d²), independent of stored item count.

```python
ResonanceStore(dim: int, *, eta: float = 1e-3, top_k: Optional[int] = None)

store.store(vector: NDArray, label: Optional[int] = None) → int
store.spectral_density(query: NDArray) → float   # ρ(q) via Green's function
store.retrieve(query: NDArray, top_n: int = 5) → List[Tuple[int, float]]
store.size → int          # property
store.trace_norm() → float
```

**Complexity:**
- Ingestion: O(d²) per vector (rank-1 update)
- Retrieval: O(d²) (one resolve), independent of N

---

### 5.15 Orchestrator

**Module:** `unitarity_labs.core.orchestrator`

Spectral coherence loop integrating `TransportEvaluator` and `ResonanceStore`.

```python
Orchestrator(dim: int, *, r_warn: float = 0.45, eta: float = 1e-3, top_k=None)

orch.ingest(hidden_state: NDArray) → StepRecord
orch.retrieve(query: NDArray, top_n: int = 5) → List[Tuple[int, float]]
orch.evaluate_health() → HealthReport
orch.log_session_audit(*, tag: str = "") → Path
orch.history → List[StepRecord]
```

**StepRecord:**
```python
@dataclass
class StepRecord:
    step: int
    r_ratio: Optional[float]
    spectral_density: float
    frobenius_dist: float
    warned: bool
```

**Warning:** `SymmetryBreakWarning` — issued when ⟨r⟩ < 0.45.

---

### 5.16 Validator

**Module:** `unitarity_labs.core.validator`

Benchmark comparison and text-based metric extraction (k = 1 Kar–Berry–Keating sector).

```python
# Reference benchmark
GROK_4_MARCH_2026_BENCHMARK = {"r_ratio": 0.58, "zeta": 0.85, "frobenius_stability": 0.21}

# Compare observed stats against benchmark
evaluate_model_health(input_stats: ModelStats, benchmark: Optional[dict] = None) → HealthReport

# Regex extraction from free-form text (handles Unicode, markdown, LaTeX)
parse_metrics_from_text(text: str) → ModelStats

# Write timestamped JSON audit log
log_audit(report: HealthReport, stats=None, *, log_dir=None, tag="") → Path
```

**ModelStats:**
```python
@dataclass
class ModelStats:
    r_ratio: Optional[float] = None
    zeta: Optional[float] = None
    frobenius_stability: Optional[float] = None
```

**HealthReport:**
```python
@dataclass
class HealthReport:
    delta_r: Optional[float] = None
    delta_zeta: Optional[float] = None
    delta_frobenius: Optional[float] = None
    spectral_divergence: float = 0.0    # Euclidean distance in (⟨r⟩, ζ, F) space
    passed: bool = True                  # True if divergence ≤ 1.0
    details: str = ""
```

---

### 5.17 GUELoss

**Module:** `unitarity_labs.core.gue_loss`

Differentiable fine-tuning objective for GUE spectral rigidity. Uses Hutchinson's trace estimator (O(k·d²)) instead of full eigendecomposition (O(d³)).

```python
GUELoss(target_r: float = 0.578, n_vectors: int = 8)

loss_fn.forward(matrix: Tensor) → Tuple[Tensor, float]  # (loss, r_measured)
```

---

### 5.18 DiversitySnapshotMonitor

**Module:** `unitarity_labs.core.diversity_snapshot`

Drift detection via periodic solo inference windows. Every 4096 tokens, disables the bridge for 128 tokens and compares solo vs bridged hidden states.

```python
DiversitySnapshotMonitor(
    interval_tokens: int = 4096,
    solo_window_tokens: int = 128,
    collapse_threshold: float = 0.08,
    enabled: bool = True,
    auto_reanchor: bool = False,
)

monitor.step() → None                           # Call after each token
monitor.record_states(h_solo, h_bridged) → None # Capture during solo window
monitor.in_solo_window → bool
monitor.should_disable_bridge → bool
monitor.checkpoints → List[SnapshotCheckpoint]
monitor.collapse_warning_count → int
monitor.reanchor_requested → bool
monitor.clear_reanchor_request() → None
```

**Trigger condition:** If `mean(ΔH) < 0.08 × ‖H‖_F` for 2 consecutive checkpoints → "coherence collapse" warning.

---

### 5.19 HeartbeatDashboard

**Module:** `unitarity_labs.core.dashboard`

Rich-powered terminal dashboard displaying live metrics.

```python
HeartbeatDashboard(wrapper: UniversalHookWrapper, refresh_rate: float = 0.5)

dashboard.run_once() → None   # Print single snapshot
dashboard.run() → None        # Blocking live-update loop (Ctrl+C to stop)
```

Displays: ζ, spectral gap, flux epsilon, VRAM usage, head allocation, mirror state.

---

## 6. Distributed API Reference

### 6.1 DualNodeEntanglementBridge

**Module:** `unitarity_labs.core.dual_link`

ZeroMQ-based cross-process coordination for dual-node (A↔B) inference.

```python
DualNodeEntanglementBridge(
    node_id: str = "A",
    krylov_dim: int = 128,
    zmq_port: int = 5555,
)

link.send_krylov_basis(krylov_basis: Tensor) → None
link.recv_partner_basis(device: str = "cpu") → Optional[Tensor]
link.compute_cross_sync(my_basis, partner_basis) → float  # phi_AB
link.unitary_rotation_inject(h_local, partner_basis, phi_AB) → Tensor
link.close() → None
```

**Register non-invasively:**
```python
register_dual_node_hook(bridge: CrossLayerEntanglementHook, node_id: str) → Callable
```

**Ports:** Node A publishes on 5555, subscribes on 5556. Node B is reversed.

---

### 6.2 Handshake Protocol

**Module:** `unitarity_labs.core.handshake`

Byzantine-resistant ZMQ handshake with precision exchange, epoch negotiation, and nonce agreement.

```python
perform_handshake(
    pub_socket, sub_socket,
    local_node_id: str,
    local_precision: PrecisionClass,
    local_epoch_len: int = 16,
    timeout_ms: int = 5000,
    capability_proxy: Optional[float] = None,
) → Dict[str, Any]
# Returns: {remote_id, remote_precision, epoch_len, nonce, my_nonce,
#           remote_capability, remote_tps_estimate, remote_clock_offset}

validate_precision_pair(local: PrecisionClass, remote: PrecisionClass) → bool
compute_capability_ratio(local_cap, remote_cap) → float
```

**Exceptions:**
- `IncompatibleNode` — raised when precision pair is invalid
- `HandshakeTimeout` — raised on timeout

---

### 6.3 ByzantineVoting

**Module:** `unitarity_labs.core.kill_switch`

Leaderless Byzantine fault detection with β_TB thresholds.

```python
ByzantineVoting(max_faulty: int = 1)

voting.get_status(node_id: str) → NodeStatus
voting.report_beta(node_id: str, beta: float) → NodeStatus
voting.suspect(suspect_id, accuser_id, reason="") → bool
voting.cast_ban_vote(suspect_id, voter_id) → None
voting.quorum_check(suspect_id) → bool
voting.is_influence_nullified(node_id) → bool
voting.desync_sever(node_id, accuser_id) → NodeStatus
voting.set_observer(node_id) → NodeStatus
voting.quarantine_node(node_id, accuser_id) → bool
voting.evaluate_bridge_state(node_id, beta_tb, local_node_id)
    → Tuple[NodeStatus, bool]  # (status, should_sever)
```

**NodeStatus enum:** `ACTIVE`, `DEGRADED`, `SEVERED`, `BANNED`, `OBSERVER`

**β_TB thresholds:**

| Threshold | Value | Action |
|-----------|-------|--------|
| Hard sever | < 0.20 | Immediate disconnect |
| Graceful | < 0.35 | Graceful degradation |
| Readmit | ≥ 0.45 | Node readmission |

---

### 6.4 PrecisionProjector

**Module:** `unitarity_labs.core.precision_projector`

Handles precision mismatch between nodes.

```python
class PrecisionClass(str, Enum):
    INT4 = "INT4"
    FP8_E4M3 = "FP8_E4M3"
    BF16 = "BF16"
    FP32 = "FP32"

CANONICAL_DTYPE = torch.bfloat16  # All cross-node messages cast to this

DequantAdapter(dim: int, bias: bool = True)
adapter.forward(x: Tensor) → Tensor

add_dither(x: Tensor, bits: int = 16) → Tensor  # Stochastic rounding noise
get_projector(src: PrecisionClass, tgt: PrecisionClass, dim: int) → Optional[DequantAdapter]
has_projector(src: PrecisionClass, tgt: PrecisionClass) → bool
```

---

### 6.5 SemanticLock

**Module:** `unitarity_labs.core.semantic_lock`

Non-local semantic locking with multi-round nonce commitment, anchor consensus, and erasure-coded shard distribution.

#### NonceCommitProtocol

```python
NonceCommitProtocol(local_node_id: str, max_faulty: int = 1, session_uuid: str = "")

protocol.generate_commit() → Tuple[str, str]          # (node_id, commit_hash)
protocol.receive_commit(node_id, commit_hash) → None
protocol.check_commit_quorum() → bool
protocol.reveal_nonce() → Tuple[str, bytes]
protocol.receive_reveal(node_id, nonce) → bool
protocol.check_reveal_quorum() → bool
protocol.finalize_nonce(timestamp_floor=0) → bytes     # Final shared nonce
```

#### SemanticLockController

```python
SemanticLockController(local_node_id, max_faulty=1, session_uuid="", dim=64)

controller.initialize_from_nonce(nonce: bytes) → None
controller.step(h_layer7, h_layer12, U_base) → Tuple[Tensor, float, bool]
# Returns: (modulated_output, alpha_sem, byzantine_flag)
controller.get_shards() → List[Tensor]
controller.drain_accusations() → List[Tuple[str, float]]
controller.compute_alpha(h_layer7, h_layer12) → float
controller.initialized → bool
controller.v21_fallback_active → bool
```

#### Functions

```python
semantic_anchor_init(nonce: bytes, dim: int = 64) → Tuple[Tensor, Tensor]  # (W_sem, anchor_k)
compute_alpha_sem(h_layer, anchor_k, W_sem) → float
compute_alpha_sem_ensemble(h_layer7, h_layer12, anchor_k, W_sem, ...) → float
semantic_modulation(U_base, alpha_sem_final, bridge_strength) → Tensor
holographic_semantic_shard_encode(W_sem, anchor_k, redundancy=4) → List[Tensor]
holographic_semantic_shard_decode(shards, dim=64, redundancy=4) → Tuple[Tensor, Tensor, bool]
validate_shard_integrity(shards, dim=64) → Tuple[bool, str]
```

**Constants:**
- `SEM_PROJECTION_DIM = 64`
- `ANCHOR_DRIFT_THRESHOLD = 0.08`
- `ALPHA_SEM_FULL_BRIDGE = 0.85`
- `ALPHA_SEM_PARTIAL_FLOOR = 0.30`
- `ALPHA_SEM_BYZANTINE_THRESHOLD = 0.15`
- `ERASURE_REDUNDANCY = 4`
- `ANCHOR_GOSSIP_INTERVAL = 1024`
- `ANCHOR_FREEZE_TOKENS = 256`

---

### 6.6 ChronosLock

**Module:** `unitarity_labs.core.chronos_lock`

Temporal synchronization for distributed nodes with TPS estimation, desync monitoring, and Reed-Solomon encoded shards.

```python
ChronosLock(node_id: str)

lock.update_tps(measured_tps: float) → float           # Updated TPS EMA
lock.update_desync(Δτ: float, num_nodes: int = 2) → bool  # Sever threshold?
lock.record_τ(τ_value: float) → None
lock.compute_τ_hash() → Optional[str]                  # SHA-256
lock.validate_τ_chain(received_prev_hash) → bool
lock.handle_jump(expected_seq, received_seq) → Tuple[bool, bool]
lock.unitary_wait_spin(h: Tensor, t_wait: float, mode="auto") → Tensor
lock.check_probation(Δτ: float) → bool
lock.is_on_probation() → bool
lock.encode_shard() → bytes                             # RS-encoded
lock.decode_shard(raw: bytes) → Tuple[int, float, float, Optional[str], float]
lock.should_timestamp_sync() → bool
lock.prepare_timestamp_msg() → dict
lock.apply_timestamp_responses(responses: dict) → float  # Clock offset
```

---

### 6.7 TierManager

**Module:** `unitarity_labs.dist.tier_manager`

Manages tier assignment for distributed nodes based on TPS attestation.

```python
class NodeTier(Enum):
    COMPUTE = "COMPUTE"   # TPS ≥ threshold
    ROUTER = "ROUTER"     # TPS below threshold

TierManager(min_compute_tps: float = 10.0, max_wait_before_demotion: float = 2.0)

manager.attest(node_id, tps_ema, tps_variance) → NodeTier
manager.record_wait(node_id, wait_secs) → None  # Demotes to ROUTER on excess wait
manager.get_record(node_id) → _NodeRecord
```

---

### 6.8 KillSwitch

See [ByzantineVoting](#63-byzantinevoting) — same module (`unitarity_labs.core.kill_switch`).

---

## 7. Labs (Experimental)

**Package:** `unitarity_labs.labs`

> **WARNING:** Modules in this package are research prototypes. APIs may change without notice.

### topology_metrics

```python
from unitarity_labs.labs.topology_metrics import (
    spectral_gap_from_activations,
    betti_0_from_weights,
    activation_entropy_profile,
)

spectral_gap_from_activations(activations: Dict[int, Tensor]) → Dict[int, float]
betti_0_from_weights(model: nn.Module, threshold: float = 0.1) → Dict[str, int]
activation_entropy_profile(activations: Dict[int, Tensor]) → Dict[int, float]
```

### Re-export Shims

| Module | Re-exports from |
|--------|----------------|
| `unitarity_labs.labs.flux` | `unitarity_labs.core.flux` |
| `unitarity_labs.labs.mirror` | `unitarity_labs.core.mirror` |
| `unitarity_labs.labs.semantic_lock` | `unitarity_labs.core.semantic_lock` |

---

## 8. Benchmarks

All benchmarks share a common harness with standard CLI arguments:

```
--mode {passive,active}   Runtime mode (default: active)
--seed INT                Random seed (default: 42)
--output PATH             Write JSON results to file
```

### GSM8K (`benchmarks/gsm8k.py`)

Grade-school math reasoning. Measures manifold coherence (ζ), baseline cosine, and permutation p-value alongside accuracy and latency.

### HumanEval+ (`benchmarks/humaneval_plus.py`)

Code generation benchmark. Compares passive vs active mode code completion accuracy with spectral diagnostics.

### Agent Instruct (`benchmarks/agent_instruct.py`)

Instruction-following benchmark. Evaluates how spectral alignment affects instruction adherence.

### Adversarial Safety (`benchmarks/adversarial_safety.py`)

Adversarial robustness testing. Measures whether active-mode intervention affects safety refusal rates.

### Result Format

```json
{
  "results": [
    {
      "zeta": 0.847123,
      "baseline_cosine": 0.312456,
      "permutation_p": 0.001,
      "latency_ms": 45.2,
      "accuracy": 0.85
    }
  ]
}
```

---

## 9. Constants & Thresholds Reference

### Bridge & Flux

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `STAGGER_FRACTION` | 0.25 | flux | Fraction of heads entangled per step |
| `HAWKING_DECAY_RATE` | 0.95 | flux | Flux epsilon decay rate |
| `TAYLOR_DIM_THRESHOLD` | 64 | flux | Switch to Taylor expm above this |
| `TAYLOR_ERROR_GUARD` | 1e-6 | flux | Taylor series error bound |
| `PROJECTION_NORM_MIN` | 0.01 | bridge | LoRA projection norm floor |
| `PROJECTION_NORM_MAX` | 10.0 | bridge | LoRA projection norm ceiling |

### Mirror

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `DEFAULT_ALPHA` | 0.1 | mirror | Proprioceptive injection strength |
| `CATASTROPHE_ALPHA` | 0.3 | mirror | Injection abort threshold |
| `NUM_METRIC_CHANNELS` | 4 | mirror | Metrics injected at layer 0 |
| `HOLOGRAPHIC_SAFETY_FACTOR` | 100.0 | mirror | Safety margin for bit rate |

### Spectral

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `GOE_R_MEAN` | 0.5307 | spectral_monitor | GOE expected ⟨r⟩ |
| `GUE_R_MEAN` | 0.5996 | spectral_monitor | GUE expected ⟨r⟩ |
| `POISSON_R_MEAN` | 0.3863 | spectral_monitor | Poisson expected ⟨r⟩ |
| `R_RATIO_FLOOR` | 0.40 | spectral_monitor | Below = manifold collapse |
| `COHERENCE_R_WARN` | 0.45 | orchestrator | SymmetryBreakWarning threshold |

### Byzantine

| Threshold | Value | Module | Description |
|-----------|-------|--------|-------------|
| Hard sever | < 0.20 | kill_switch | Immediate disconnect |
| Graceful degrade | < 0.35 | kill_switch | Enter DEGRADED state |
| Readmit | ≥ 0.45 | kill_switch | Allow node re-entry |

### Semantic Lock

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `SEM_PROJECTION_DIM` | 64 | semantic_lock | Semantic projection dimension |
| `ANCHOR_DRIFT_THRESHOLD` | 0.08 | semantic_lock | Anchor drift alert |
| `ALPHA_SEM_FULL_BRIDGE` | 0.85 | semantic_lock | Full bridge α_sem |
| `ALPHA_SEM_PARTIAL_FLOOR` | 0.30 | semantic_lock | Partial bridge floor |
| `ALPHA_SEM_BYZANTINE_THRESHOLD` | 0.15 | semantic_lock | Byzantine flag trigger |
| `ERASURE_REDUNDANCY` | 4 | semantic_lock | Reed-Solomon redundancy |
| `ANCHOR_GOSSIP_INTERVAL` | 1024 | semantic_lock | Tokens between gossip |
| `ANCHOR_FREEZE_TOKENS` | 256 | semantic_lock | Freeze after re-anchor |

### Diversity Snapshot

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `SNAPSHOT_INTERVAL_TOKENS` | 4096 | diversity_snapshot | Drift check interval |
| `SOLO_WINDOW_TOKENS` | 128 | diversity_snapshot | Solo window length |
| `COLLAPSE_THRESHOLD_RATIO` | 0.08 | diversity_snapshot | ΔH/‖H‖ trigger |
| `CONSECUTIVE_TRIGGERS_REQUIRED` | 2 | diversity_snapshot | Consecutive lows needed |

### Benchmark

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `r_ratio` | 0.58 | validator | Reference GUE gap ratio |
| `zeta` | 0.85 | validator | Reference manifold coherence |
| `frobenius_stability` | 0.21 | validator | Reference ‖A−I‖_F |

---

## 10. Google Colab Guide

### Install (v3.1.6+)

```python
# Cell 1 — Install
!pip install unitarity-labs
```

### Verify Installation

```python
# Cell 2 — Verify
from unitarity_labs.core import __version__
print(f"unitarity-labs {__version__}")
# Expected: 3.1.6-Singularity
```

### Passive Observation (no GPU mutation)

```python
# Cell 3 — Passive mode (CPU-safe, no GPU required)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

wrapper = UniversalHookWrapper(model, config=model.config, mode="passive")

inputs = tokenizer("The fundamental theorem of arithmetic states that", return_tensors="pt")
with torch.no_grad():
    wrapper(**inputs)

metrics = wrapper.get_metrics()
for k, v in metrics.items():
    print(f"  {k}: {v}")
```

### Active Mode with ζ Comparison

```python
# Cell 4 — Active mode ζ comparison (CPU-safe)
wrapper_active = UniversalHookWrapper(model, config=model.config, mode="active")
wrapper_active.ensure_device()

test_prompts = {
    "LOGIC": "Every integer greater than 1 is a prime or a product of primes.",
    "CHAOS": "purple triangle desk jump oxygen 999 elephant sandwich frequency.",
}

for label, text in test_prompts.items():
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        wrapper_active(**inputs)
    zeta = wrapper_active.bridge.bell_correlation
    print(f"{label}: ζ = {zeta:.6f}")
```

### Validate External Model Output

```python
# Cell 5 — Validate metrics from text
from unitarity_labs.core.validator import parse_metrics_from_text, evaluate_model_health

text = "The model reports ⟨r⟩ = 0.55 and ζ ≈ 0.78"
stats = parse_metrics_from_text(text)
report = evaluate_model_health(stats)
print(f"Divergence: {report.spectral_divergence:.4f}")
print(f"Passed: {report.passed}")
```

### With GPU (Colab T4/A100)

> **v3.1.6 fix:** `model.device` is unreliable under `device_map="auto"` because
> HuggingFace may place layers on different devices. The code below uses
> `next(model.parameters()).device` to infer the embedding device instead.
> For T4 GPUs (compute capability < 8.0), precision auto-selects INT4/FP16
> instead of BF16 to avoid unsupported bfloat16 operations.

```python
# Cell 6 — Active mode with GPU (Colab T4/A100)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper

model_name = "gpt2"  # or "unsloth/Llama-3.2-1B-bnb-4bit" with GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

wrapper = UniversalHookWrapper(model, config=model.config, mode="active")
wrapper.ensure_device()  # Sync bridge/LoRA/mirror to the actual layer devices

# Derive input device from the model's embedding layer
# (do NOT use model.device — it is undefined under device_map="auto")
input_device = next(model.parameters()).device
inputs = tokenizer("Explain quantum entanglement.", return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"\nζ = {wrapper.bridge.bell_correlation:.6f}")
print(f"Spectral gap = {wrapper.bridge.spectral_gap():.6f}")
```

### Full Active Mode with Real Model (Colab T4)

> Use this cell for real-model inference on Colab free tier (T4 GPU).
> The T4 lacks native BF16 ALUs, so we use FP16 + 4-bit quantization.

```python
# Cell 7 — Real model on T4 (INT4 / FP16)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unitarity_labs.core.universal_hook import UniversalHookWrapper

model_id = "unsloth/Llama-3.2-1B-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

wrapper = UniversalHookWrapper(
    model=model,
    config=model.config,
    mode="active",
    flux_ratio=0.25,
)
wrapper.ensure_device()

prompt = "Explain cross-layer alignment in three sentences."
inputs = tokenizer(prompt, return_tensors="pt")
input_device = next(model.parameters()).device
inputs = {k: v.to(input_device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
print(f"\nζ = {wrapper.bridge.bell_correlation:.6f}")
print(f"Spectral gap = {wrapper.bridge.spectral_gap():.6f}")
print(f"Flux kicks = {wrapper.get_metrics()['flux_kicks_total']}")
```

### Metrics Dashboard (text output)

```python
# Cell 8 — Print full metrics snapshot
metrics = wrapper.get_metrics()
print("=" * 50)
print("UNITARITY-LAB METRICS")
print("=" * 50)
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")
```

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'unitarity_labs'`

**Cause:** Old cached version or incorrect install.

```bash
pip install --upgrade --force-reinstall unitarity-labs
```

### `ModuleNotFoundError: No module named 'start_node'`

**Cause:** You have version < 3.1.5 which had a broken console script entry point.

```bash
pip install "unitarity-labs>=3.1.6"
```

### `RuntimeError: CUDA out of memory`

Use a smaller model or force lower precision:

```python
wrapper = UniversalHookWrapper(
    model, config=model.config,
    mode="passive",           # No tensor mutation = less VRAM
    flux_ratio=0.1,           # Fewer active heads
)
```

Or use INT4 quantization:
```bash
unitarity-start --precision INT4 --model unsloth/Llama-3.2-1B-bnb-4bit
```

### `ValueError: Unsupported model architecture`

The wrapper needs to find transformer layers at `model.model.layers`, `model.layers`, or `model.transformer.h`. If your model uses a different structure, open an issue with the model class name.

### Version Mismatch Between Files

All three locations must agree:
- `unitarity_labs/core/version.py` (`__version__`)
- `pyproject.toml` (`version`)
- `setup.py` (`_META["version"]`)

Check with:
```python
from unitarity_labs.core.version import __version__
print(__version__)  # Should be "3.1.6-Singularity"
```

### Dual-Node ZMQ Timeout

Ensure both nodes run on the same network and ports 5555/5556 are open:
```bash
# Terminal 1
unitarity-start --node-id A --dual

# Terminal 2
unitarity-start --node-id B --dual
```

### Dashboard Not Showing

Install `rich` explicitly if missing:
```bash
pip install rich>=13.0
unitarity-start --dashboard
```

---

## 12. Changelog

### v3.1.6-Singularity (2026-03-18)
- **CRITICAL FIX:** Colab T4 active-mode runtime tensor placement
  - Sink hook now coerces LoRA adapter, bridge bias, and mirror submodules to the live activation device/dtype before applying them
  - `ensure_device()` derives device from the actual hooked sink layer, not `next(model.parameters())` which may return an embedding on CPU under `device_map="auto"`
  - `ProprioceptiveHook.forward` coerces metrics to `metric_proj.weight.device` (belt-and-suspenders for BNB-4bit edge cases)
  - `EigenConsciousnessIntegrator.forward` verifies hook submodule device before calling it
- **T4 BF16 guard:** GPUs with compute capability < 8.0 (T4, Turing) in the 8–24 GB VRAM tier now route to INT4/FP16 instead of BF16
- **Removed all `model.device` reliance:** Replaced with embedding-layer device inference in `start_node.py`, `cli.py`, and `run_community.py`
- `VirtualLayer13` no longer hardcodes `torch.device("cuda")`; defers to live activation device
- Fixed `setup.py` version to match `version.py`

### v3.1.5-Singularity (2026-03-18)
- **CRITICAL FIX:** Moved CLI entry points into `unitarity_labs/cli.py` for PyPI/Colab compatibility
- Console scripts (`unitarity-start`, `unitarity-validate`) now resolve correctly from pip-installed wheels
- Fixed `test_hardening.py` version assertions
- Updated `index.html` to v3.1.5-Singularity

### v3.1.4-Singularity (2026-03-18)
- Version bump for CI/CD trigger
- Clean build artifacts

### v3.1.3-Singularity (2026-03-18)
- Refactored to standard Python package structure (`unitarity_labs/` parent)
- Moved `core/`, `dist/`, `labs/` into `unitarity_labs/`
- Updated all 62+ internal imports
- `setup.py` uses `find_packages()`

### v3.1.2-Singularity
- Initial PyPI release
- Full v3.0 stack: bridge, flux governor, mirror, ghost layer, virtual layer 13
- Dual-node ZMQ coordination
- Byzantine fault tolerance
- Spectral monitoring and benchmarks

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

## Links

- **PyPI:** https://pypi.org/project/unitarity-labs/
- **GitHub:** https://github.com/holeyfield33-art/unitarity-lab
- **Website:** See `index.html` for the SHI Leaderboard and live audit tool
