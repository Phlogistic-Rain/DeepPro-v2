import os
import sys
from typing import Literal, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from iPro.models.FusionNet import FusionNet

GatingMode = Literal["static", "dynamic", "bio"]
RouterMode = Literal["softmax", "sparsemax", "entmax15"]

BIO_DIM = 20

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    K = z.shape[dim]
    rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = K
    rng = rng.view(view_shape)
    support = (1 + rng * z_sorted) > z_cumsum
    k = support.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    k_idx = (k.long() - 1).clamp(min=0)
    tau = (z_cumsum.gather(dim, k_idx) - 1) / k
    return torch.clamp(z - tau, min=0)

def _entmax15(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    try:
        from entmax import entmax15 as _e15
    except Exception as e:
        raise RuntimeError(
            "router='entmax15' requires the `entmax` package (not installed). "
            "Run `pip install entmax --no-deps`, or use router='sparsemax' instead."
        ) from e
    return _e15(z, dim=dim)

class MoEFusionNet(FusionNet):
    def __init__(
        self,
        config,
        n_train: int,
        gating: GatingMode = "static",
        gate_hidden: int = 64,
        router: str = "softmax",
        gate_temperature: float = 1.0,
        bio_hidden: int = 16,
    ):
        super().__init__(config, n_train)
        self.gating: GatingMode = gating
        self.router: str = router
        self.gate_temperature: float = float(gate_temperature)

        if gating == "static":
            self.gate_logits = nn.Parameter(torch.zeros(self.num_views))
        elif gating == "dynamic":
            self.gate_out = nn.Linear(gate_hidden, self.num_views)
            self.gate_net = nn.Sequential(
                nn.Linear(self.num_views * self.emb_dim, gate_hidden),
                nn.ReLU(),
                self.gate_out,
            )
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
        elif gating == "bio":
            self.gate_out = nn.Linear(bio_hidden, self.num_views)
            self.gate_net = nn.Sequential(
                nn.Linear(BIO_DIM, bio_hidden),
                nn.ReLU(),
                self.gate_out,
            )
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
        else:
            raise ValueError(f"Unknown gating mode: {gating}")

        self.last_gate: Optional[torch.Tensor] = None
        self._gate_live: Optional[torch.Tensor] = None

    def _normalize(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_temperature != 1.0:
            logits = logits / self.gate_temperature
        if self.router == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.router == "sparsemax":
            return sparsemax(logits, dim=-1)
        if self.router == "entmax15":
            return _entmax15(logits, dim=-1)
        raise ValueError(f"Unknown router: {self.router}")

    @staticmethod
    def bio_summary(one_hot: torch.Tensor) -> torch.Tensor:
        B, _, L = one_hot.shape
        mono = one_hot.mean(dim=2)
        left = one_hot[:, :, :-1]
        right = one_hot[:, :, 1:]
        di = torch.einsum("bil,bjl->bij", left, right)
        di = di.reshape(B, 16) / max(L - 1, 1)
        return torch.cat([mono, di], dim=1)

    def _gate(self, views: dict, bio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gating == "static":
            return self._normalize(self.gate_logits)
        if self.gating == "bio":
            if bio is None:
                raise ValueError("gating='bio' requires the bio composition summary input (see DeepProV2.forward).")
            return self._normalize(self.gate_net(bio))
        x = torch.cat([views[k] for k in range(self.num_views)], dim=-1)
        return self._normalize(self.gate_net(x))

    def fuse(self, inputs: dict, bio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self.encoder(inputs)
        stacked = torch.stack([views[k] for k in range(self.num_views)], dim=0)
        g = self._gate(views, bio)

        if self.gating == "static":
            fusion = (g.view(-1, 1, 1) * stacked).sum(dim=0)
        else:
            fusion = (g.t().unsqueeze(-1) * stacked).sum(dim=0)

        self.last_gate = g.detach()
        self._gate_live = g
        return fusion, g

    def forward(self, inputs: dict, bio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        fusion, _ = self.fuse(inputs, bio)
        logits = self.classifier(fusion)
        return logits, fusion

    def gate_reg(self) -> torch.Tensor:
        if self.gating == "static":
            g = self._normalize(self.gate_logits)
        else:
            if self.last_gate is None:
                return torch.zeros((), device=self.gate_out.bias.device)
            g = self.last_gate.mean(dim=0)
        return self._kl_to_uniform(g)

    def gate_reg_train(self) -> torch.Tensor:
        if self.gating == "static":
            return self.gate_reg()
        if self._gate_live is None:
            return torch.zeros((), device=self.gate_out.bias.device)
        return self.gate_reg_from(self._gate_live)

    def gate_reg_from(self, g: torch.Tensor) -> torch.Tensor:
        if g.dim() == 2:
            g = g.mean(dim=0)
        return self._kl_to_uniform(g)

    def _kl_to_uniform(self, g: torch.Tensor) -> torch.Tensor:
        uniform = 1.0 / self.num_views
        eps = 1e-8
        return (g * ((g + eps) / uniform).log()).sum()
