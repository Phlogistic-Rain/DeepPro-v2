from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MeltingHeadConfig:
    hidden: int = 32
    dh: int = 32
    ksize: int = 5
    length: int = 81

class MeltingHead(nn.Module):
    def __init__(self, config: MeltingHeadConfig = MeltingHeadConfig()):
        super().__init__()
        self.cfg = config
        pad = config.ksize // 2
        self.trunk = nn.Sequential(
            nn.Conv1d(4, config.hidden, config.ksize, padding=pad),
            nn.ReLU(),
            nn.Conv1d(config.hidden, config.hidden, config.ksize, padding=pad),
            nn.ReLU(),
        )
        self.rho_head = nn.Conv1d(config.hidden, 1, 1)
        self.feat = nn.Sequential(nn.Linear(config.hidden, config.dh), nn.ReLU())

    def forward(self, one_hot: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(one_hot)
        rho = torch.sigmoid(self.rho_head(h)).squeeze(1)
        pooled = h.mean(dim=-1)
        melt = self.feat(pooled)
        return {"rho": rho, "melt": melt}

    @staticmethod
    def melt_loss(rho: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(rho, target)

if __name__ == "__main__":
    torch.manual_seed(0)
    head = MeltingHead()
    x = torch.randn(8, 4, 81).softmax(1)
    out = head(x)
    print("rho:", tuple(out["rho"].shape), "melt:", tuple(out["melt"].shape))
    tgt = torch.rand(8, 81)
    print("melt_loss:", MeltingHead.melt_loss(out["rho"], tgt).item())
