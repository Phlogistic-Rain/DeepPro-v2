from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from iPro.models.DNAbert import BERT
from iPro.models.DNAbert2 import DNABERT2Encoder, MODEL_NAME as DNABERT2_MODEL
from iPro.models.NTv2 import NucleotideTransformerV2
from iPro.models.FusionNet import FusionNet

@dataclass
class DeepProConfig:
    feature_dims: List[int] = field(default_factory=lambda: [768, 768, 768, 512])

    emb_dim: int = 128
    fc_hidden: int = 256

    beta: float = 1e-3

    sigma: float = 1.0
    lambda_1: float = 1.0
    lambda_2: float = 1e-4
    b: float = 0.02

class DeepPro(nn.Module):

    VIEW_KEYS: Tuple[str, ...] = ("dnabert_3mer", "dnabert_6mer", "dnabert2", "ntv2")

    def __init__(self, config: DeepProConfig, n_train: int):
        super().__init__()
        self.config = config
        self.n_train = n_train

        self.dnabert_3mer = BERT(kmer=3)
        self.dnabert_6mer = BERT(kmer=6)
        self.dnabert2 = DNABERT2Encoder(pooling="mean", freeze=False)
        self.ntv2 = NucleotideTransformerV2(freeze=False)

        self.FusionNet = FusionNet(config, n_train)

    @staticmethod
    def _move(data: Dict[str, Dict[str, torch.Tensor]], device) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            view: {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for view, batch in data.items()
        }

    def extract_features(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        device = next(self.parameters()).device
        data = self._move(data, device)

        feats: Dict[int, torch.Tensor] = {}

        b3 = data["dnabert_3mer"]
        feats[0] = self.dnabert_3mer(
            input_ids=b3["input_ids"],
            attention_mask=b3["attention_mask"],
            token_type_ids=b3["token_type_ids"],
        )

        b6 = data["dnabert_6mer"]
        feats[1] = self.dnabert_6mer(
            input_ids=b6["input_ids"],
            attention_mask=b6["attention_mask"],
            token_type_ids=b6["token_type_ids"],
        )

        b2 = data["dnabert2"]
        feats[2] = self.dnabert2(
            input_ids=b2["input_ids"],
            attention_mask=b2["attention_mask"],
        )

        nt = data["ntv2"]
        feats[3] = self.ntv2(
            input_ids=nt["input_ids"],
            attention_mask=nt["attention_mask"],
            return_mean_pooled=True,
        )["mean_embeddings"]

        return feats

    def pre_train(self, data, label: torch.Tensor, idx: torch.Tensor):
        with torch.no_grad():
            feats = self.extract_features(data)
        return self.FusionNet.dec_part(feats, label, idx)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(data)
        return self.FusionNet(feats)

    def freeze_backbones(self) -> None:
        for m in (self.dnabert_3mer, self.dnabert_6mer, self.dnabert2, self.ntv2):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def unfreeze_backbones(self) -> None:
        for m in (self.dnabert_3mer, self.dnabert_6mer, self.dnabert2, self.ntv2):
            for p in m.parameters():
                p.requires_grad = True

def build_tokenizers():
    from transformers import BertTokenizerFast, AutoTokenizer

    from iPro.models.DNAbert import DNABERT_KMER_SOURCE

    NTV2_MODEL = NucleotideTransformerV2.DEFAULT_MODEL_NAME
    return {
        "dnabert_3mer": BertTokenizerFast.from_pretrained(DNABERT_KMER_SOURCE[3]),
        "dnabert_6mer": BertTokenizerFast.from_pretrained(DNABERT_KMER_SOURCE[6]),
        "dnabert2": AutoTokenizer.from_pretrained(DNABERT2_MODEL, trust_remote_code=True, use_fast=True),
        "ntv2": AutoTokenizer.from_pretrained(NTV2_MODEL, trust_remote_code=True, use_fast=True),
    }
