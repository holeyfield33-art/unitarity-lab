"""
universal_hook.py — Universal Hugging Face Wrapper (v1.8)
=========================================================
Safe, non-invasive hooks for any ``AutoModelForCausalLM``.

v1.8 features:
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

from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .bridge import CrossLayerEntanglementHook
from .dual_link import register_dual_node_hook
from .precision_projector import PrecisionClass
from .handshake import perform_handshake, validate_precision_pair, IncompatibleNode
from .ghost_layer import RecursiveMirror


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
        flux_ratio: float = 0.25,
        head_rotate_steps: int = 50,
        precision: PrecisionClass = PrecisionClass.BF16,
        initial_epoch_len: int = 16,
        reorth_interval: int = 256,
    ):
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
        )

        # Head-staggering mask
        self.head_mask = torch.zeros(self.num_heads, dtype=torch.bool)
        self._rotate_heads()
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
            for idx in (self.mid_idx, self.last_idx):
                self.layers[idx].register_forward_hook(
                    partial(self.dual_hook, layer_idx=idx)
                )

        self.step_counter = 0

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
    # Forward pass
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Forward pass through the wrapped model.

        Hooks fire automatically via PyTorch; head mask rotates every
        ``head_rotate_steps`` steps.
        """
        self.step_counter += 1
        if self.step_counter % self.head_rotate_steps == 0:
            self._rotate_heads()
        # v2.0: Periodic re-orthogonalization from bridge
        if self.step_counter % self.reorth_interval == 0:
            self.bridge.reorthogonalize()
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Monitoring API (heartbeat dashboard)
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, object]:
        """Collect current bridge/flux metrics for the dashboard."""
        metrics = {
            "bell_correlation": self.bridge.bell_correlation,
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
        """Remove all hooks registered by the bridge."""
        self.bridge.remove_hooks()
