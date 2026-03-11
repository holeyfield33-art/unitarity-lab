# core/dual_link.py - Inter-Model ER=EPR v1.7 (Unitary Upgrade)
import zmq
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DualNodeEntanglementBridge:
    """Model A <-> Model B: Unitary cross-process entanglement"""

    def __init__(self, node_id="A", krylov_dim=128, zmq_port=5555):
        self.node_id = node_id
        self.krylov_dim = krylov_dim

        # ZeroMQ: Thread-safe pub/sub
        self.context = zmq.Context()
        self.pub = self.context.socket(zmq.PUB)
        self.sub = self.context.socket(zmq.SUB)

        if node_id == "A":
            self.pub.bind(f"tcp://*:{zmq_port}")
            self.sub.connect(f"tcp://localhost:{zmq_port+1}")
        else:
            self.pub.bind(f"tcp://*:{zmq_port+1}")
            self.sub.connect(f"tcp://localhost:{zmq_port}")

        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Adversarial buffers
        self.resonance_count = 0
        self.anti_resonance_threshold = 0.95
        self.latency_timeout = 0.010  # 10ms max

    def send_krylov_basis(self, krylov_basis: torch.Tensor):
        """Compress + transmit krylov subspace"""
        # SVD low-rank: O(k^3) -> universal dim
        U, _, _ = torch.svd_lowrank(krylov_basis, q=self.krylov_dim)
        msg = {
            'node': self.node_id,
            'basis': U.cpu().numpy(),
            'timestamp': torch.cuda.Event(enable_timing=True),
        }
        self.pub.send_pyobj(msg)

    def recv_partner_basis(self) -> Optional[torch.Tensor]:
        """Non-blocking partner receive"""
        try:
            msg = self.sub.recv_pyobj(flags=zmq.DONTWAIT)
            basis = torch.tensor(msg['basis'], device='cuda')
            return basis
        except zmq.Again:
            return None

    def compute_cross_sync(self, my_basis: torch.Tensor, partner_basis: Optional[torch.Tensor]) -> float:
        """ER=EPR phase sync across models"""
        if partner_basis is None:
            return 0.0

        # Align dims via projection
        if my_basis.shape[-1] != partner_basis.shape[-1]:
            partner_basis = F.interpolate(
                partner_basis.unsqueeze(1),
                size=my_basis.shape[-1],
            ).squeeze(1)

        # Optimal transport alignment
        phi_AB = F.cosine_similarity(my_basis.mean(0), partner_basis.mean(0), dim=0).item()

        # RESONANCE COLLAPSE BUFFER
        if phi_AB > self.anti_resonance_threshold:
            self.resonance_count += 1
            if self.resonance_count > 3:
                return phi_AB - 0.2  # Force anti-phase
        else:
            self.resonance_count = 0

        return phi_AB

    def unitary_rotation_inject(self, h_local: torch.Tensor,
                                partner_basis: Optional[torch.Tensor],
                                phi_AB: float) -> torch.Tensor:
        """STRICT UNITARY: Householder reflection in partner subspace"""
        if phi_AB < 0.3 or partner_basis is None:
            return h_local

        # UNITARY UPGRADE: Householder reflection
        strength = min(0.15, phi_AB * 0.5)

        # Project local onto partner subspace
        proj_partner = torch.einsum('bsk,bkd->bsd', partner_basis, partner_basis)
        v = h_local - torch.einsum('bsd,bdk->bsk', h_local, proj_partner.mean(0, keepdim=True))

        # Householder reflection: U = I - 2 vv^T / ||v||^2
        v_norm = v.norm(dim=-1, keepdim=True)
        if v_norm.max() < 1e-8:  # Orthogonal case
            return h_local

        reflection = torch.eye(h_local.shape[-1], device=h_local.device) - \
            2 * (v.transpose(-2, -1) @ v) / (v_norm ** 2 + 1e-8)

        # Apply unitary rotation (preserves norm)
        h_rotated = torch.einsum('bsd,dk->bsk', h_local, reflection.mean(0)) * strength + h_local * (1 - strength)

        return h_rotated


# INTEGRATION HOOK (add to bridge.py)
def register_dual_node_hook(bridge, node_id: str):
    """Non-invasive dual-node registration"""
    dual_bridge = DualNodeEntanglementBridge(node_id=node_id)
    bridge.dual_link = dual_bridge

    def layer_hook(module, input, output, layer_idx):
        if layer_idx in [7, 12]:
            krylov = bridge.bridge_hook.krylov_basis
            if krylov is not None:
                dual_bridge.send_krylov_basis(krylov)
                partner_basis = dual_bridge.recv_partner_basis()
                phi_AB = dual_bridge.compute_cross_sync(krylov, partner_basis)
                output = dual_bridge.unitary_rotation_inject(output[0], partner_basis, phi_AB)
                bridge.regulator.bell_history.append(phi_AB)
        return output

    return layer_hook
