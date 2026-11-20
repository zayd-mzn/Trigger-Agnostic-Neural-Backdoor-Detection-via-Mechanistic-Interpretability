"""SST-2 loading and tokenisation helpers."""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

BASE_MODEL = "google/bert_uncased_L-4_H-256_A-4"  # 11M params, CPU friendly
MAX_LEN = 64


def load_sst2(n_train: int | None = None, n_val: int | None = None, seed: int = 0):
    """Return (train_texts, train_labels, val_texts, val_labels).

    GLUE's SST-2 test split has no labels, so the validation split (872 sentences)
    is used as the held-out test set. A disjoint slice of train is used for the
    clean reference set that the detector sees.
    """
    ds = load_dataset("nyu-mll/glue", "sst2")
    tr = ds["train"].shuffle(seed=seed)
    if n_train:
        tr = tr.select(range(n_train))
    va = ds["validation"]
    if n_val:
        va = va.select(range(min(n_val, len(va))))
    return tr["sentence"], tr["label"], va["sentence"], va["label"]


def get_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def encode(texts, tokenizer, max_len: int = MAX_LEN):
    return tokenizer(list(texts), padding="max_length", truncation=True,
                     max_length=max_len, return_tensors="pt")


def make_loader(texts, labels, tokenizer, batch_size=64, shuffle=False):
    enc = encode(texts, tokenizer)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(list(labels)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
