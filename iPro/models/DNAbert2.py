from typing import Literal, Optional

import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

MODEL_NAME = "zhihan1996/DNABERT-2-117M"

def _disable_triton_flash_attn() -> None:
    for name, mod in list(sys.modules.items()):
        if name.endswith(".bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None

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

class DNABERT2Encoder(nn.Module):

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pooling: Literal["mean", "max", "cls"] = "mean",
        freeze: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        config = BertConfig.from_pretrained(model_name)

        self.backbone = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True,
        )
        _disable_triton_flash_attn()
        self.hidden_size = config.hidden_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

        return _masked_pool(hidden, attention_mask, self.pooling)

def build_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
