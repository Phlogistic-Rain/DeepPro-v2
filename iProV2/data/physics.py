from __future__ import annotations

import numpy as np

__all__ = ["BASES", "one_hot_encode", "one_hot_batch", "dg_profile", "melting_target", "melting_target_batch"]

BASES = "ACGT"
_BASE_IDX = {b: i for i, b in enumerate(BASES)}

_SANTALUCIA_DG = {
    "AA": -1.00, "AC": -1.44, "AG": -1.28, "AT": -0.88,
    "CA": -1.45, "CC": -1.84, "CG": -2.17, "CT": -1.28,
    "GA": -1.30, "GC": -2.24, "GG": -1.84, "GT": -1.44,
    "TA": -0.58, "TC": -1.30, "TG": -1.45, "TT": -1.00,
}

def one_hot_encode(seq: str, length: int = 81) -> np.ndarray:
    x = np.zeros((4, length), dtype=np.float32)
    for i, ch in enumerate(seq[:length]):
        j = _BASE_IDX.get(ch.upper())
        if j is not None:
            x[j, i] = 1.0
    return x

def one_hot_batch(seqs, length: int = 81) -> np.ndarray:
    return np.stack([one_hot_encode(s, length) for s in seqs], axis=0)

def dg_profile(seq: str, length: int = 81) -> np.ndarray:
    s = seq[:length].upper()
    n = len(s)
    stack = np.zeros(max(n - 1, 0), dtype=np.float32)
    for j in range(n - 1):
        stack[j] = _SANTALUCIA_DG.get(s[j:j + 2], 0.0)

    pos = np.zeros(length, dtype=np.float32)
    for i in range(min(n, length)):
        left = stack[i - 1] if i - 1 >= 0 and i - 1 < len(stack) else None
        right = stack[i] if i < len(stack) else None
        vals = [v for v in (left, right) if v is not None]
        pos[i] = float(np.mean(vals)) if vals else 0.0
    return pos

def melting_target(seq: str, length: int = 81) -> np.ndarray:
    dg = dg_profile(seq, length)
    lo, hi = float(dg.min()), float(dg.max())
    if hi - lo < 1e-6:
        return np.full(length, 0.5, dtype=np.float32)
    return ((dg - lo) / (hi - lo)).astype(np.float32)

def melting_target_batch(seqs, length: int = 81) -> np.ndarray:
    return np.stack([melting_target(s, length) for s in seqs], axis=0)
