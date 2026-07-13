from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class NucleotideTransformerV2(nn.Module):

    DEFAULT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

    def __init__(
        self,
        model_name: Optional[str] = None,
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.to(self.device_)

        self.hidden_size: int = self.model.config.hidden_size
        self.max_length: int = self.tokenizer.model_max_length

        self._backbone_frozen: bool = False
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.model.eval()
        return self

    def tokenize(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.max_length

        encoded = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device_)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device_)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_mean_pooled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        input_ids = input_ids.to(self.device_)
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
        attention_mask = attention_mask.to(self.device_)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        token_embeddings: torch.Tensor = outputs["hidden_states"][-1]

        result: Dict[str, torch.Tensor] = {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
        }

        if return_mean_pooled:
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            sum_emb = (token_embeddings * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            result["mean_embeddings"] = sum_emb / denom

        return result
