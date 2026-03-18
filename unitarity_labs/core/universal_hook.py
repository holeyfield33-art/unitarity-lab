"""
universal_hook.py — Universal Hugging Face Wrapper (v3.0.0-Singularity)
========================================================================
Safe, non-invasive hooks for any ``AutoModelForCausalLM``.

Modes
-----
  - **passive**: Hooks capture metrics (ζ, spectral gap, VRAM) without
    mutating any tensor. Bridge bias injection, LoRA mutation, flux
    kicks, and mirror injection are all disabled.
  - **active**: Full intervention — bridge bias, LoRA, flux governor,
    and mirror injection operate as in previous releases.

Key features:
  - **Staggered Flux Guard**: 25% of attention heads entangled per step,
    rotating every ``head_rotate_steps`` to cap VRAM and FLOPs.
  - **Portable layer discovery**: ``model.model.layers``, ``model.layers``,
    ``model.transformer.h`` — works on Llama, Mistral, Gemma, DeepSeek-V3.
  - **Non-invasive**: hooks via ``register_forward_hook`` only; the
    original model class is never modified.
  - **Dual-node opportunistic linking**: non-blocking ZeroMQ receive
    when ``enable_dual=True``.
  - **Resource monitoring**: ``get_metrics()`` / ``get_vram_usage()`` for
    external heartbeat dashboards.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bridge import CrossLayerEntanglementHook
from .dual_link import register_dual_node_hook
from .precision_projector import PrecisionClass
from .handshake import perform_handshake, validate_precision_pair, IncompatibleNode
from .ghost_layer import RecursiveMirror
from .virtual_layer13 import VirtualLayer13
from .safety_head import SafetyHead

_VALID_MODES = ("passive", "active")
_log = logging.getLogger(__name__)


class UniversalHookWrapper:
    """Wraps a Hugging Face causal LM and attaches wormhole bridge hooks.

    Parameters
    ----------
    model : nn.Module
        A ``transformers.AutoModelForCausalLM`` (or compatible) instance.
    config : object
        The model's config (``model.config``).
    node_id : str
        ``"A"`` or ``"B"`` for dual-node mode.
    enable_dual : bool
        Attach ZeroMQ dual-node hook when True.
    mode : str
        ``"passive"`` — hooks capture metrics only, no tensor mutation.
        ``"active"``  — full bridge intervention (default).
    flux_ratio : float
        Fraction of heads actively entangled (default 0.25).
    head_rotate_steps : int
        Steps between head-mask rotations (default 50).
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        node_id: str = "A",
        enable_dual: bool = False,
        mode: str = "active",
        flux_ratio: float = 0.25,
        head_rotate_steps: int = 50,
        precision: PrecisionClass = PrecisionClass.BF16,
        initial_epoch_len: int = 16,
        reorth_interval: int = 256,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        _log.info("UniversalHookWrapper mode=%s", mode)
        self.model = model
        self.config = config
        self.enable_dual = enable_dual
        self.flux_ratio = flux_ratio
        self.head_rotate_steps = head_rotate_steps
        self.precision = precision
        self.initial_epoch_len = initial_epoch_len
        self.reorth_interval = reorth_interval

        # Discover transformer layers
        self.layers = self._get_layers()
        self.num_layers = len(self.layers)
        self.mid_idx = self.num_layers // 2
        self.last_idx = max(0, self.num_layers - 2)

        # Determine hidden dim and num_heads from config
        self.num_heads = self._resolve_num_heads()
        self.hidden_dim = self._resolve_hidden_dim()

        # Build the CrossLayerEntanglementHook (the existing bridge)
        self.bridge = CrossLayerEntanglementHook(
            model=model,
            source_layer=self.mid_idx,
            sink_layer=self.last_idx,
            num_heads=self.num_heads,
            layer_accessor=lambda _m: self.layers,
            d_model=self.hidden_dim,
        )

        # Passive mode: disable bridge tensor mutation while still
        # capturing source activations for metric computation.
        if self.mode == "passive":
            self.bridge.enabled = False

        # Head-staggering mask
        self.head_mask = torch.zeros(self.num_heads, dtype=torch.bool)
        self._rotate_heads()
        if hasattr(self.bridge, "set_head_mask"):
            self.bridge.set_head_mask(self.head_mask)

        # v2.1: Recursive Mirror with Schism Hardening
        self.recursive_mirror = RecursiveMirror(bridge=self.bridge, config=config)

        # Optional dual-node link
        self.dual_hook = None
        if enable_dual:
            self.dual_hook = register_dual_node_hook(self.bridge, node_id=node_id)
            # Set precision and epoch_len on the dual link
            if hasattr(self.bridge, 'dual_link') and self.bridge.dual_link is not None:
                self.bridge.dual_link.precision = precision
                self.bridge.dual_link.epoch_len = initial_epoch_len
                self.bridge.dual_link._reorth_interval = reorth_interval
                # v3.0: Attach VirtualLayer13 + SafetyHead to dual link
                self.bridge.dual_link.attach_virtual_layer13(config, node_id)
            for idx in (self.mid_idx, self.last_idx):
                self.layers[idx].register_forward_hook(
                    partial(self.dual_hook, layer_idx=idx)
                )

        self.step_counter = 0
        self._step_hook_handle = None
        self._register_step_hook()
        self.ensure_device()

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------
    def _get_layers(self) -> nn.ModuleList:
        """Locate the list of transformer blocks in any HF model."""
        layers: object
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):  # type: ignore[union-attr]
            layers = self.model.model.layers  # type: ignore[union-attr]
        elif hasattr(self.model, "layers"):
            layers = self.model.layers  # type: ignore[union-attr]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):  # type: ignore[union-attr]
            layers = self.model.transformer.h  # type: ignore[union-attr]
        else:
            raise ValueError(
                "Unsupported model architecture: cannot locate transformer layers. "
                "Please open an issue with your model class."
            )
        return layers  # type: ignore[return-value]

    def _resolve_num_heads(self) -> int:
        for attr in ("num_attention_heads", "num_heads", "n_head"):
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        return 32  # safe fallback

    def _resolve_hidden_dim(self) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        return 4096

    # ------------------------------------------------------------------
    # Head staggering
    # ------------------------------------------------------------------
    def _rotate_heads(self) -> None:
        """Randomly select a new subset of heads (flux_ratio fraction)."""
        k = max(1, int(self.flux_ratio * self.num_heads))
        indices = torch.randperm(self.num_heads)[:k]
        new_mask = torch.zeros(self.num_heads, dtype=torch.bool)
        new_mask[indices] = True
        self.head_mask = new_mask
        if hasattr(self.bridge, "set_head_mask"):
            self.bridge.set_head_mask(self.head_mask)

    # ------------------------------------------------------------------
    # Step hook registration
    # ------------------------------------------------------------------
    def _register_step_hook(self) -> None:
        """Register a forward hook on layer 0 to track steps.

        This ensures the step counter, head rotation, and
        re-orthogonalization fire on every forward pass — including
        when called via model.generate(), which bypasses __call__.
        """
        def _step_hook(_module, _input, _output):
            self.step_counter += 1
            if self.mode == "active":
                if self.step_counter % self.head_rotate_steps == 0:
                    self._rotate_heads()
                if self.step_counter % self.reorth_interval == 0:
                    self.bridge.reorthogonalize()

        handle = self.layers[0].register_forward_hook(_step_hook)
        self._step_hook_handle = handle

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Forward pass through the wrapped model.

        Hooks fire automatically via PyTorch; step tracking, head
        rotation, and re-orthogonalization are handled by the step
        hook on layer 0, so they run whether this method or
        model.generate() is the caller.
        """
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Monitoring API (heartbeat dashboard)
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, object]:
        """Collect current bridge/flux metrics for the dashboard."""
        metrics = {
            "mode": self.mode,
            "manifold_coherence_zeta": self.bridge.bell_correlation,
            "bell_correlation": self.bridge.bell_correlation,  # back-compat alias
            "spectral_gap": self.bridge.spectral_gap(),
            "flux_epsilon": self.bridge.flux_governor.epsilon,
            "flux_kicks_total": len(self.bridge.flux_governor.kick_history),
            "flux_stagnation": self.bridge.flux_governor.stagnation_count,
            "bridge_enabled": self.bridge.enabled,
            "active_heads": int(self.head_mask.sum().item()),
            "total_heads": self.num_heads,
            "step": self.step_counter,
        }
        # v2.1: Recursive mirror diagnostics
        if hasattr(self, 'recursive_mirror'):
            mirror = self.recursive_mirror
            metrics["mirror_depth"] = mirror.target_layer
            metrics["mirror_quarantined"] = list(mirror.quarantine)
            metrics["mirror_accusations"] = len(mirror.accusations)
        return metrics

    def get_vram_usage(self) -> Tuple[int, int]:
        """Return ``(used_mib, total_mib)`` via ``pynvml`` if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used // 1024**2, info.total // 1024**2
        except Exception:
            return (0, 0)

    def remove_hooks(self) -> None:
        """Remove all hooks registered by the bridge and step tracker."""
        self.bridge.remove_hooks()
        if self._step_hook_handle is not None:
            self._step_hook_handle.remove()
            self._step_hook_handle = None

    def ensure_device(self) -> None:
        """Move all bridge sub-modules to the model's device and dtype.

        Call this after construction when the model is on CUDA.
        Ensures LoRA adapter, mirror, gate, and proprioceptive hook
        tensors match the model's device to prevent cross-device errors
        during model.generate().

        When ``device_map="auto"`` is active, ``next(model.parameters())``
        may return an embedding on CPU while transformer blocks live on
        CUDA.  We therefore derive the authoritative device from the
        *sink layer* that actually hosts the hook — this is the device
        the sink activation will arrive on at runtime.
        """
        # --- Derive device from the actual hooked sink layer, not the
        #     global model parameter iterator (safe under device_map="auto")
        try:
            device = next(self.layers[self.last_idx].parameters()).device
        except StopIteration:
            device = next(self.model.parameters()).device
        try:
            dtype = next(self.layers[self.last_idx].parameters()).dtype
        except StopIteration:
            dtype = next(self.model.parameters()).dtype

        # Set device reference on bridge
        if hasattr(self.bridge, '_device'):
            self.bridge._device = device

        # LoRA adapter (lora_A, lora_B, lora_B_projection)
        if hasattr(self.bridge, 'lora_adapter'):
            self.bridge.lora_adapter.to(device=device, dtype=dtype)

        # EigenConsciousnessIntegrator (ProprioceptiveHook + TopologicalGate)
        if hasattr(self.bridge, 'mirror'):
            self.bridge.mirror.to(device=device, dtype=dtype)

        # RecursiveMirror
        if hasattr(self, 'recursive_mirror') and hasattr(self.recursive_mirror, 'to'):
            self.recursive_mirror.to(device=device, dtype=dtype)

        # Head mask
        if hasattr(self, 'head_mask') and self.head_mask is not None:
            self.head_mask = self.head_mask.to(device=device)

        # Clear cached activations that may be on a stale device
        if hasattr(self.bridge, '_source_activation') and self.bridge._source_activation is not None:
            self.bridge._source_activation = self.bridge._source_activation.to(device)
        if hasattr(self.bridge, '_bridge_eigenvectors') and self.bridge._bridge_eigenvectors is not None:
            self.bridge._bridge_eigenvectors = self.bridge._bridge_eigenvectors.to(device)
        if hasattr(self.bridge, '_bridge_bias') and self.bridge._bridge_bias is not None:
            self.bridge._bridge_bias = self.bridge._bridge_bias.to(device)

        _log.info("Bridge components moved to %s / %s", device, dtype)

    # ── Geometric Brain Buffer ──────────────────────────────────────────
    # Stores post-MLP hidden states for spectral rigidity analysis.
    # See tests/test_geometric_rigidity.py and GEOMETRIC_BRAIN.md

    _instance = None  # Class-level reference for test access

    def register_geometric_hooks(self, layers: list):
        """
        Registers forward hooks to capture post-MLP residual stream.
        Target: layer.mlp output at each specified layer index.
        Minimum S=512 tokens required for valid GUE measurement.
        """
        if not hasattr(self, '_buffer'):
            self._buffer = {}

        UniversalHookWrapper._instance = self

        for idx in layers:
            try:
                target = list(self.model.model.layers)[idx].mlp
                target.register_forward_hook(self._hook_fn(idx))
            except (AttributeError, IndexError) as e:
                print(f"[GeometricBrain] Could not hook layer {idx}: {e}")

    def _hook_fn(self, layer_idx: int):
        """Returns a hook function that stores detached activations."""
        def hook(module, input, output):
            # Detach + clone prevents compute graph memory leaks
            if not hasattr(self, '_buffer'):
                self._buffer = {}
            self._buffer[layer_idx] = output.detach().clone()
        return hook

    def get_buffer(self, layer: int):
        """
        Exposes manifold sample for r-ratio audit.
        Returns tensor of shape [batch, seq, dim].
        Raises ValueError if hook not registered or inference not run.
        """
        if not hasattr(self, '_buffer') or layer not in self._buffer:
            raise ValueError(
                f"No manifold data at layer {layer}. "
                "Call register_geometric_hooks() and run inference first."
            )
        return self._buffer[layer]

    def clear_buffer(self):
        """Frees buffer memory between audit runs."""
        self._buffer = {}
