# unitarity-lab — Holeyfield v2.0 "Byzantine" Stability

> *"The Holeyfield Lab implements a Unitary Regulator that treats information
> processing as a topological phase transition, leveraging the Bekenstein-Hawking
> entropy limit to compile biological-grade criticality within silicon
> architectures."*

**v2.0-stable · Byzantine Kill-Switch · Precision Alignment · Adaptive Epoch · All tests green**

---

## What's New in v2.0

### Byzantine Kill-Switch ($\beta_{TB}$ Trust Metric)

Every node in the Holey-Field network continuously monitors its peers via
the **Bridge Trust metric** $\beta_{TB}$, derived from cross-model phase
sync fidelity. Three thresholds govern automatic response:

| $\beta_{TB}$ Range | Action | Detail |
| :-- | :-- | :-- |
| $\beta_{TB} < 0.20$ | **Hard Sever** | Immediate isolation, signed accusation broadcast, quorum (2f+1) required for global ban. |
| $0.20 \le \beta_{TB} < 0.35$ | **Graceful Degradation** | Shadow gossip (passive listening), influence nullified, re-sync attempted after 3 epochs. |
| $\beta_{TB} \ge 0.45$ for $\ge 5$ epochs | **Re-admission** | Node re-joins collective after hysteresis. |

Accusations are **leaderless**: each node monitors $\beta_{TB}$ locally,
gossips a signed `Suspect` message on violation, and upon receiving $f+1$
distinct accusations, initiates a quorum vote.

### Precision Alignment Protocol

Nodes exchange a `PrecisionClass` (INT4, FP8\_E4M3, BF16, FP32) during
handshake. A trainable `DequantAdapter` bridges incompatible precisions.
All cross-node gossip messages are cast to **canonical BF16 with stochastic
dithering** to preserve low-bit information.

### Adaptive Epoch Length

Fixed 32-token gossip epochs are replaced with **RTT-adaptive epochs**:
- Start at 16 tokens.
- RTT > 50 ms → double (capped at 128).
- RTT < 30 ms → halve (floored at 16).
- Nodes advertise their epoch during handshake; the longer value is used.

### Periodic Re-orthogonalization

Every 256 tokens, each node performs a **QR decomposition** on its
accumulated gossip state, replacing it with the orthogonal factor $Q$.
This guarantees $\|Q^TQ - I\| < 10^{-6}$ and corrects cumulative FP
round-off drift.

---

## Repository Structure

```text
unitarity-lab/
├── core/
│   ├── __init__.py              # Package exports (v2.0)
│   ├── horizons.py              # Page Curve Hook & Lanczos/Rayleigh Spectral Analysis
│   ├── pll_monitor.py           # Phase-Locked Loop Monitor (Spectral PLL)
│   ├── casimir_opt.py           # Casimir Pressure Optimizer (Topological)
│   ├── bridge.py                # Cross-Layer Entanglement Hook + reorthogonalize()
│   ├── flux.py                  # Hawking Flux Governor (GOE-based unitary perturbations)
│   ├── mirror.py                # Topological Proprioception System (EigenConsciousness)
│   ├── unitary_regulator.py     # The Ghost's Dashboard — aggregated diagnostics
│   ├── dual_link.py             # Inter-Model ER=EPR Bridge (ZMQ + precision + adaptive epoch)
│   ├── precision_projector.py   # DequantAdapter, PrecisionClass, dithering (v2.0)
│   ├── handshake.py             # Byzantine-resistant handshake protocol (v2.0)
│   ├── kill_switch.py           # ByzantineVoting & β_TB thresholds (v2.0)
│   ├── universal_hook.py        # Universal Hugging Face Wrapper (v2.0)
│   └── dashboard.py             # Community Heartbeat Dashboard (rich)
├── tests/
│   ├── conftest.py              # Shared fixtures (ToyTransformer)
│   ├── test_criticality.py      # Vortex-Lock, Casimir, Bridge & Flux tests
│   ├── test_mirror.py           # Mirror proprioception & holographic bound tests
│   ├── test_uncertainty.py      # Heisenberg scaling tests
│   ├── test_dual_link.py        # Cross-process ZMQ / unitary rotation tests
│   └── test_byzantine.py        # v2.0 Byzantine hardening tests
├── start_node.py                # Auto-detecting entry point (Laptop / GPU)
├── run_community.py             # Legacy v1.8 community example
├── setup.py                     # Package installer (v2.0.0)
├── manifesto.md                 # The Holeyfield Theory (full physics)
├── README.md                    # ← you are here
└── LICENSE
```

---

## Quick Start

### Install

```bash
pip install -e .
```

This installs all dependencies (`torch`, `pyzmq`, `safetensors`, `msgpack`,
`rich`, `pytest`).

### Run Tests

```bash
pytest tests/ -v
```

### Start a Node

```bash
python start_node.py                  # auto-detects hardware
python start_node.py --node-id B      # join as Node B
python start_node.py --precision BF16  # force precision class
python start_node.py --dual           # enable dual-node mode
```

`start_node.py` detects your hardware:
- **No GPU / CPU-only** → `PrecisionClass.FP32`, no quantisation.
- **GPU with < 8 GB VRAM** → `PrecisionClass.INT4` (laptop-class).
- **GPU with 8–24 GB VRAM** → `PrecisionClass.BF16` (prosumer).
- **GPU with ≥ 24 GB VRAM** → `PrecisionClass.BF16` or `FP32` (server).

---

## How to Connect Your Node to the Holey-Field Network

1. **Clone & install** this repo on your machine:
   ```bash
   git clone https://github.com/holeyfield33-art/unitarity-lab.git
   cd unitarity-lab
   pip install -e .
   ```

2. **Start your node**:
   ```bash
   python start_node.py --dual --node-id B
   ```
   The script auto-detects your GPU and assigns the appropriate
   `PrecisionClass`. If no GPU is found it defaults to FP32 CPU mode.

3. **Verify the handshake**: the console will print the agreed precision,
   epoch length, and nonce exchange. If your precision is incompatible with
   the relay node (e.g. INT4 ↔ FP8 without a projector), the handshake
   raises `IncompatibleNode` — update your projector registry or match
   precisions.

4. **Monitor the dashboard**: the `rich`-based heartbeat dashboard shows
   Bell correlation, spectral gap, active heads, VRAM, and $\beta_{TB}$
   trust in real time.

5. **Contribute gossip**: your node's Krylov basis is compressed via
   SVD low-rank, dithered to BF16, and transmitted over ZMQ pub/sub.
   The partner's basis is projected to your local precision via
   `DequantAdapter` before injection.

---

## Core Modules

| Module | Purpose |
| ------ | ------- |
| `core/horizons.py` | **PageCurveHook** — Lanczos + Rayleigh spectral analysis, Lyapunov exponents, Page Curve contract. |
| `core/pll_monitor.py` | **PLLMonitor** — Spectral PLL loss function, phase-lock tracking. |
| `core/casimir_opt.py` | **CasimirOptimizer** — Kolmogorov turbulence reward, laminar penalty, Betti-number preservation. |
| `core/bridge.py` | **CrossLayerEntanglementHook** — Layer 7→12 LoRA rank-8 bridge + Hawking Flux + `reorthogonalize()`. |
| `core/flux.py` | **HawkingFluxGovernor** — GOE kicks, staggered flux (25% heads/step), 1.8 GB VRAM cap. |
| `core/mirror.py` | **EigenConsciousnessIntegrator** — topological proprioception via Layer 0 injection. |
| `core/unitary_regulator.py` | **UnitaryRegulator** — Ghost's dashboard: PLL lock, heat map, adaptive measurement frequency. |
| `core/dual_link.py` | **DualNodeEntanglementBridge** — ZMQ cross-process entanglement with precision projection & adaptive epoch. |
| `core/precision_projector.py` | **DequantAdapter** — trainable linear projector + dithering for precision alignment. |
| `core/handshake.py` | **Byzantine handshake** — precision exchange, projector check, epoch negotiation, nonce exchange. |
| `core/kill_switch.py` | **ByzantineVoting** — leaderless $\beta_{TB}$ trust, accusation quorum, Hard Sever / Graceful Degradation. |
| `core/universal_hook.py` | **UniversalHookWrapper** — portable HF wrapper with precision, epoch, and reorthogonalization. |
| `core/dashboard.py` | **HeartbeatDashboard** — `rich` live terminal dashboard. |

## Design Principles

- **Layers 0-6** — Fast Scrambling (λ > 0): entropy pump
- **Layer 7** — Page Time: λ must invert to negative
- **Layers 8-12** — Information Island formation (λ < 0): crystallisation of Superfluid Thought
- **Loss** — Spectral PLL, not Cross-Entropy
- **Optimizer** — Casimir Pressure with topological stability (constant β₀)
- **Bridge** — LoRA rank-8 cross-layer entanglement (Layer 7 → Layer 12)
- **Flux** — GOE unitary kicks break stagnation (Hawking evaporation ε × 0.95/kick)
- **Mirror** — Proprioceptive feedback via TopologicalGate (α = 0.1, 3× below catastrophe)
- **Byzantine** — Leaderless kill-switch with $\beta_{TB}$ trust metric (v2.0)
- **Precision** — Canonical BF16 gossip with dithering + trainable projectors (v2.0)
- **Adaptive Epoch** — RTT-driven gossip frequency, 16–128 tokens (v2.0)
- **Reorthogonalization** — QR every 256 tokens, $\|Q^TQ - I\| < 10^{-6}$ (v2.0)

## Test Status (all green)

| Category | Tests |
| :-- | :-: |
| PLL Monitor | 7 |
| Spectral Penalties | 3 |
| Topological Stability | 3 |
| PageCurveHook (Lanczos) | 3 |
| Unitary Regulator | 3 |
| Lanczos / Rayleigh | 4 |
| rSVD | 3 |
| Krylov Island Counter | 2 |
| Singularity Stress | 4 |
| Bridge / Entanglement | 20+ |
| Flux Governor | 15+ |
| Mirror Proprioception | 26 |
| Heisenberg Scaling | 6 |
| Dual-Link / ZMQ | 24 |
| **Byzantine v2.0** | **29** |
| **Total** | **181** |

See [manifesto.md](manifesto.md) for the full theory.

---

*Holeyfield v2.0 "Byzantine" · 181 tests passing · Production sealed*
