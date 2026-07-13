import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from iPro.models.DNAbert import BERT
from iPro.models.DNAbert2 import DNABERT2Encoder
from iPro.models.NTv2 import NucleotideTransformerV2
from iPro.models.FusionNet import FusionNet
from iPro.models.deepPro import DeepProConfig, build_tokenizers as _build_tokenizers_v1

from iProV2.models.ProkBERT import ProkBERTEncoder
from iProV2.models.moe_fusion import MoEFusionNet, GatingMode
from iProV2.models.grammar_moe import GrammarMoE, GrammarMoEConfig

@dataclass
class DeepProV2Config(DeepProConfig):
    feature_dims: List[int] = field(default_factory=lambda: [768, 768, 768, 512, 384])
    ntv2_model_name: Optional[str] = None

class DeepProV2(nn.Module):

    VIEW_KEYS: Tuple[str, ...] = ("dnabert_3mer", "dnabert_6mer", "dnabert2", "ntv2", "prokbert")

    def __init__(
        self,
        config: DeepProV2Config,
        n_train: int,
        use_moe: bool = True,
        gating: GatingMode = "static",
        use_grammar: bool = True,
        grammar_cfg: Optional[GrammarMoEConfig] = None,
        router: str = "softmax",
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.n_train = n_train

        self.dnabert_3mer = BERT(kmer=3)
        self.dnabert_6mer = BERT(kmer=6)
        self.dnabert2 = DNABERT2Encoder(pooling="mean", freeze=False)
        self.ntv2 = NucleotideTransformerV2(model_name=config.ntv2_model_name, freeze=False)
        self.prokbert = ProkBERTEncoder(pooling="mean", freeze=False)

        self._backbones = (
            self.dnabert_3mer, self.dnabert_6mer, self.dnabert2, self.ntv2, self.prokbert,
        )

        self.use_moe = use_moe
        if use_moe:
            self.FusionNet: FusionNet = MoEFusionNet(
                config, n_train, gating=gating, router=router, gate_temperature=gate_temperature,
            )
        else:
            self.FusionNet = FusionNet(config, n_train)

        self.use_grammar = use_grammar
        if use_grammar:
            gcfg = grammar_cfg or GrammarMoEConfig()
            self.grammar = GrammarMoE(gcfg)
            self.cls_head = nn.Sequential(
                nn.Linear(config.emb_dim + gcfg.dm, 20), nn.Dropout(0.5), nn.ReLU(), nn.Linear(20, 2),
            )

    @staticmethod
    def _move(data: Dict[str, Dict[str, torch.Tensor]], device) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            view: {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for view, batch in data.items()
            if view in DeepProV2.VIEW_KEYS
        }

    def extract_features(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        device = next(self.parameters()).device
        data = self._move(data, device)

        feats: Dict[int, torch.Tensor] = {}

        b3 = data["dnabert_3mer"]
        feats[0] = self.dnabert_3mer(
            input_ids=b3["input_ids"], attention_mask=b3["attention_mask"],
            token_type_ids=b3["token_type_ids"],
        )

        b6 = data["dnabert_6mer"]
        feats[1] = self.dnabert_6mer(
            input_ids=b6["input_ids"], attention_mask=b6["attention_mask"],
            token_type_ids=b6["token_type_ids"],
        )

        b2 = data["dnabert2"]
        feats[2] = self.dnabert2(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"])

        nt = data["ntv2"]
        feats[3] = self.ntv2(
            input_ids=nt["input_ids"], attention_mask=nt["attention_mask"],
            return_mean_pooled=True,
        )["mean_embeddings"]

        pb = data["prokbert"]
        feats[4] = self.prokbert(
            input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
            token_type_ids=pb["token_type_ids"],
        )

        return feats

    def pre_train(self, data, label: torch.Tensor, idx: torch.Tensor):
        with torch.no_grad():
            feats = self.extract_features(data)
        return self.FusionNet.dec_part(feats, label, idx)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(data)

        bio = None
        fn = self.FusionNet
        if isinstance(fn, MoEFusionNet) and fn.gating == "bio":
            oh = data["grammar"]["onehot"].to(next(self.parameters()).device, non_blocking=True)
            bio = fn.bio_summary(oh)

        if not self.use_grammar:
            if isinstance(fn, MoEFusionNet):
                return fn(feats, bio)
            return fn(feats)

        if isinstance(fn, MoEFusionNet):
            rep, _ = fn.fuse(feats, bio)
        else:
            _, rep = fn(feats)

        one_hot = data["grammar"]["onehot"].to(rep.device, non_blocking=True)
        m = self.grammar(one_hot)["m"]
        logits = self.cls_head(torch.cat([rep, m], dim=-1))
        return logits, rep

    def freeze_backbones(self) -> None:
        for m in self._backbones:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def unfreeze_backbones(self) -> None:
        for m in self._backbones:
            for p in m.parameters():
                p.requires_grad = True

def build_tokenizers_v2():
    from iProV2.models.ProkBERT import build_tokenizer as build_prokbert_tokenizer
    toks = _build_tokenizers_v1()
    toks["prokbert"] = build_prokbert_tokenizer()
    return toks
