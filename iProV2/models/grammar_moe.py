from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GrammarMoEConfig:
    K: int = 6
    w35: int = 6
    w10: int = 6
    d_min: int = 10
    d_max: int = 25
    d_init: float = 17.0
    d_init_sd: float = 3.0
    dm: int = 64
    length: int = 81
    c35: float = 30.0
    c10: float = 49.0
    pos_sd: float = 7.0
    pos_weight: float = 1.0
    hetero: bool = False
    n_sig54: int = 1
    c35_sig54: float = 36.0
    c10_sig54: float = 48.0
    d_init_sig54: float = 12.0

class GrammarMoE(nn.Module):
    def __init__(self, config: GrammarMoEConfig = GrammarMoEConfig()):
        super().__init__()
        self.cfg = config
        K, w35, w10 = config.K, config.w35, config.w10

        self.P35 = nn.Parameter(torch.randn(K, 4, w35) * 0.1)
        self.P10 = nn.Parameter(torch.randn(K, 4, w10) * 0.1)

        d_grid = torch.arange(config.d_min, config.d_max + 1, dtype=torch.float32)
        log_gauss = -0.5 * ((d_grid - config.d_init) / config.d_init_sd) ** 2
        log_phi = log_gauss.unsqueeze(0).repeat(K, 1).clone()
        if config.hetero and config.n_sig54 > 0:
            log_g54 = -0.5 * ((d_grid - config.d_init_sig54) / config.d_init_sd) ** 2
            log_phi[K - config.n_sig54:] = log_g54.unsqueeze(0)
        self.log_phi = nn.Parameter(log_phi)

        self._F = 5
        self.proj = nn.Linear(self._F, config.dm)

        Lout = config.length - config.w35 + 1
        i_grid = torch.arange(Lout, dtype=torch.float32)
        self.register_buffer("i_grid", i_grid)
        if config.hetero:
            mu35 = torch.full((K,), config.c35)
            mu10 = torch.full((K,), config.c10)
            if config.n_sig54 > 0:
                mu35[K - config.n_sig54:] = config.c35_sig54
                mu10[K - config.n_sig54:] = config.c10_sig54
            self.mu35 = nn.Parameter(mu35)
            self.mu10 = nn.Parameter(mu10)
        else:
            self.register_buffer("pos35_prior",
                                 config.pos_weight * (-0.5 * ((i_grid - config.c35) / config.pos_sd) ** 2))
            self.register_buffer("pos10_prior",
                                 config.pos_weight * (-0.5 * ((i_grid - config.c10) / config.pos_sd) ** 2))

    def _scan(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logP35 = F.log_softmax(self.P35, dim=1)
        logP10 = F.log_softmax(self.P10, dim=1)
        s35 = F.conv1d(one_hot, logP35)
        s10 = F.conv1d(one_hot, logP10)
        L = min(s35.shape[-1], s10.shape[-1])
        return s35[..., :L], s10[..., :L]

    def _position_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.hetero:
            i = self.i_grid.unsqueeze(0)
            p35 = cfg.pos_weight * (-0.5 * ((i - self.mu35.unsqueeze(1)) / cfg.pos_sd) ** 2)
            p10 = cfg.pos_weight * (-0.5 * ((i - self.mu10.unsqueeze(1)) / cfg.pos_sd) ** 2)
            return p35, p10
        return self.pos35_prior.unsqueeze(0), self.pos10_prior.unsqueeze(0)

    def _soft_align(self, s35: torch.Tensor, s10: torch.Tensor):
        cfg = self.cfg
        _, K, L = s35.shape
        p35_prior, p10_prior = self._position_priors()
        Kp = p35_prior.shape[0]
        terms = []
        d_ids = []
        j_starts = []
        for d in range(cfg.d_min, cfg.d_max + 1):
            if d >= L:
                break
            di = d - cfg.d_min
            valid = L - d
            block = (s35[..., :valid]
                     + s10[..., d:d + valid]
                     + self.log_phi[:, di].view(1, K, 1)
                     + p35_prior[:, :valid].view(1, Kp, valid)
                     + p10_prior[:, d:d + valid].view(1, Kp, valid))
            terms.append(block)
            d_ids.append(torch.full((valid,), float(d), device=s35.device))
            j_starts.append(torch.arange(d, d + valid, device=s35.device, dtype=torch.float32))

        cat = torch.cat(terms, dim=-1)
        s_e = torch.logsumexp(cat, dim=-1)

        w = torch.softmax(cat, dim=-1)
        j_vec = torch.cat(j_starts).view(1, 1, -1)
        d_vec = torch.cat(d_ids).view(1, 1, -1)
        exp_pos10 = (w * j_vec).sum(-1)
        exp_d = (w * d_vec).sum(-1)
        return s_e, exp_pos10, exp_d

    def _expert_descriptor(self, s_e, s35, s10):
        feats = torch.stack([
            s_e,
            s35.amax(-1), s35.mean(-1),
            s10.amax(-1), s10.mean(-1),
        ], dim=-1)
        return feats

    def forward(self, one_hot: torch.Tensor) -> Dict[str, torch.Tensor]:
        s35, s10 = self._scan(one_hot)
        s_e, exp_pos10, exp_d = self._soft_align(s35, s10)

        g = torch.softmax(s_e, dim=-1)

        desc = self._expert_descriptor(s_e, s35, s10)
        h = self.proj(desc)
        m = (g.unsqueeze(-1) * h).sum(dim=1)

        return {
            "m": m,
            "g": g,
            "s_e": s_e,
            "exp_pos10": exp_pos10,
            "exp_d": exp_d,
        }

    def load_balance_loss(self, g: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        mean_g = g.mean(dim=0)
        ent_mean = -(mean_g * (mean_g + eps).log()).sum()
        ent_sample = -(g * (g + eps).log()).sum(-1).mean()
        return -ent_mean + 0.1 * ent_sample

    def expert_diversity_loss(self) -> torch.Tensor:
        K = self.cfg.K
        if K < 2:
            return torch.zeros((), device=self.P10.device)
        loss = torch.zeros((), device=self.P10.device)
        for P in (self.P35, self.P10):
            p = torch.softmax(P, dim=1) - 0.25
            v = F.normalize(p.reshape(K, -1), dim=1)
            sim = v @ v.t()
            off = (sim.sum() - sim.diagonal().sum()) / (K * (K - 1))
            loss = loss + off
        return loss

    def pwm_sharpness_loss(self) -> torch.Tensor:
        eps = 1e-9
        loss = torch.zeros((), device=self.P35.device)
        for P in (self.P35, self.P10):
            p = torch.softmax(P, dim=1)
            ent = -(p * (p + eps).log()).sum(dim=1)
            loss = loss + ent.mean()
        return loss

    def pwm_probs(self) -> Dict[str, torch.Tensor]:
        return {
            "P35": torch.softmax(self.P35, dim=1).detach(),
            "P10": torch.softmax(self.P10, dim=1).detach(),
            "log_phi": self.log_phi.detach(),
        }
