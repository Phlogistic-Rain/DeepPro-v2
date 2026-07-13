import os

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel

DNABERT_KMER_SOURCE = {
    3: os.environ.get("DNABERT_3MER", "zhihan1996/DNA_bert_3"),
    4: os.environ.get("DNABERT_4MER", "zhihan1996/DNA_bert_4"),
    5: os.environ.get("DNABERT_5MER", "zhihan1996/DNA_bert_5"),
    6: os.environ.get("DNABERT_6MER", "zhihan1996/DNA_bert_6"),
}

class BERT(nn.Module):
    def __init__(self, kmer):
        super(BERT, self).__init__()
        self.kmer = kmer

        self.pretrainpath = DNABERT_KMER_SOURCE[self.kmer]

        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,
            finetuning_task="dnaprom",
            cache_dir=None,
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, input_ids, attention_mask, token_type_ids):

        representation = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )["pooler_output"]

        return representation

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"trainable: {name}, shape: {param.shape}")

        print(f"\ntrainable params: {trainable_params:,}")
        print(f"total params: {all_param:,}")
        print(f"trainable ratio: {100 * trainable_params / all_param:.2f}%")
