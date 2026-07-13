import os
import sys
from typing import Dict, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from iProV2.models.deeppro_v2 import DeepProV2, build_tokenizers_v2
from iProV2.data.physics import one_hot_batch

DEFAULT_MAX_LEN: Dict[str, int] = {
    "dnabert_3mer": 82,
    "dnabert_6mer": 80,
    "dnabert2": 48,
    "ntv2": 20,
    "prokbert": 84,
}

def _kmer_split(seq: str, k: int) -> str:
    return " ".join(seq[i:i + k] for i in range(len(seq) + 1 - k))

class TextDatasetV2(Dataset):

    VIEW_KEYS = DeepProV2.VIEW_KEYS

    def __init__(self, csv_path: str, max_len: Optional[Dict[str, int]] = None):
        self.csv_path = csv_path
        self.max_len = dict(max_len or DEFAULT_MAX_LEN)

        df = pd.read_csv(csv_path)
        seqs = df["text"].tolist()
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        self.onehot = torch.from_numpy(one_hot_batch(seqs, 81))

        tokenizers = build_tokenizers_v2()
        tag = os.path.basename(csv_path)
        self.tokenized = {
            "dnabert_3mer": self._tokenize_kmer(
                seqs, tokenizers["dnabert_3mer"], k=3, max_len=self.max_len["dnabert_3mer"],
                desc=f"tokenize dnabert_3mer [{tag}]",
            ),
            "dnabert_6mer": self._tokenize_kmer(
                seqs, tokenizers["dnabert_6mer"], k=6, max_len=self.max_len["dnabert_6mer"],
                desc=f"tokenize dnabert_6mer [{tag}]",
            ),
            "dnabert2": self._tokenize_plain(
                seqs, tokenizers["dnabert2"], max_len=self.max_len["dnabert2"],
                desc=f"tokenize dnabert2     [{tag}]",
            ),
            "ntv2": self._tokenize_plain(
                seqs, tokenizers["ntv2"], max_len=self.max_len["ntv2"],
                desc=f"tokenize ntv2         [{tag}]",
            ),
            "prokbert": self._tokenize_plain(
                seqs, tokenizers["prokbert"], max_len=self.max_len["prokbert"],
                keep_token_type=True,
                desc=f"tokenize prokbert     [{tag}]",
            ),
        }

    @staticmethod
    def _tokenize_kmer(seqs, tokenizer, k: int, max_len: int, desc: str = "") -> Dict[str, torch.Tensor]:
        kmer_strs = [_kmer_split(s, k) for s in seqs]
        return TextDatasetV2._batch_tokenize(
            kmer_strs, tokenizer, max_len,
            keep=("input_ids", "attention_mask", "token_type_ids"),
            desc=desc,
        )

    @staticmethod
    def _tokenize_plain(
        seqs, tokenizer, max_len: int, keep_token_type: bool = False, desc: str = "",
    ) -> Dict[str, torch.Tensor]:
        keep: Tuple[str, ...] = ("input_ids", "attention_mask")
        if keep_token_type:
            keep = keep + ("token_type_ids",)
        return TextDatasetV2._batch_tokenize(
            seqs, tokenizer, max_len, keep=keep, desc=desc,
            return_token_type_ids=keep_token_type,
        )

    @staticmethod
    def _batch_tokenize(
        texts, tokenizer, max_len: int, keep: Tuple[str, ...],
        desc: str = "", chunk: int = 1000, return_token_type_ids: bool = False,
    ) -> Dict[str, torch.Tensor]:
        buffers: Dict[str, list] = {k: [] for k in keep}
        starts = list(range(0, len(texts), chunk))
        for i in tqdm(starts, desc=desc, unit="chunk", disable=not sys.stderr.isatty()):
            enc = tokenizer(
                texts[i:i + chunk], return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_len,
                return_token_type_ids=True if return_token_type_ids else None,
            )
            for k in keep:
                buffers[k].append(enc[k])
        return {k: torch.cat(v, dim=0) for k, v in buffers.items()}

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        data = {
            view: {k: v[idx] for k, v in batch.items()}
            for view, batch in self.tokenized.items()
        }
        data["grammar"] = {"onehot": self.onehot[idx]}
        return data, self.labels[idx], idx
