from typing import Literal, Optional, cast

import torch
import torch.nn as nn

from prokbert.models import ProkBertModel, ProkBertConfig

MODEL_NAME = "neuralbioinfo/prokbert-mini-c"
PoolMode = Literal["mean", "max", "cls"]

def _masked_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: Literal["mean", "max", "cls"] = "mean",
) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0, :]

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

    if mode == "mean":
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    if mode == "max":
        hidden = hidden.masked_fill(mask == 0, float("-inf"))
        return hidden.max(dim=1).values

    raise ValueError(f"Unknown pooling mode: {mode}")

class ProkBERTEncoder(nn.Module):

    DEFAULT_MODEL_NAME = MODEL_NAME

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pooling: PoolMode = "mean",
        freeze: bool = False,
    ):
        super().__init__()
        self.pooling: PoolMode = pooling
        self.model_name = model_name

        config = ProkBertConfig.from_pretrained(model_name)
        self.backbone = cast(ProkBertModel, ProkBertModel.from_pretrained(model_name, config=config))

        self.hidden_size: int = config.hidden_size
        self.max_length: int = 84

        self._backbone_frozen: bool = False
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state
        return _masked_pool(hidden, attention_mask, self.pooling)

def build_tokenizer(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
