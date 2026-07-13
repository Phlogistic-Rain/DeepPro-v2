import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
INNER = os.path.join(ROOT, 'iProV2')
for _p in (ROOT, INNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iProV2.models.deeppro_v2 import DeepProV2, DeepProV2Config, build_tokenizers_v2
from iProV2.data.physics import one_hot_batch

import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASES = 'ACGT'
L = 81
TSS = 60
SPECIES = 10
DATA_DIR = os.path.join(ROOT, 'data', 'Benchmark Dataset', 'csv')
RUN_DIR = os.path.join(INNER, 'models_log', 'DeepPro-v2', '2026-07-13_11-48')
EMB_KEY = 'FusionNet.emb.weight'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAXLEN = {"dnabert_3mer": 82, "dnabert_6mer": 80, "dnabert2": 48, "ntv2": 20, "prokbert": 84}
TOK = None

def _kmer(s, k):
    return " ".join(s[i:i + k] for i in range(len(s) + 1 - k))

def tokenize(seqs):
    e3 = TOK['dnabert_3mer']([_kmer(s, 3) for s in seqs], return_tensors='pt',
                             padding='max_length', truncation=True, max_length=MAXLEN['dnabert_3mer'])
    e6 = TOK['dnabert_6mer']([_kmer(s, 6) for s in seqs], return_tensors='pt',
                             padding='max_length', truncation=True, max_length=MAXLEN['dnabert_6mer'])
    e2 = TOK['dnabert2'](list(seqs), return_tensors='pt',
                         padding='max_length', truncation=True, max_length=MAXLEN['dnabert2'])
    en = TOK['ntv2'](list(seqs), return_tensors='pt',
                     padding='max_length', truncation=True, max_length=MAXLEN['ntv2'])
    ep = TOK['prokbert'](list(seqs), return_tensors='pt', padding='max_length',
                         truncation=True, max_length=MAXLEN['prokbert'], return_token_type_ids=True)
    oh = torch.from_numpy(one_hot_batch(list(seqs), L))
    return {
        "dnabert_3mer": {k: e3[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')},
        "dnabert_6mer": {k: e6[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')},
        "dnabert2": {k: e2[k] for k in ('input_ids', 'attention_mask')},
        "ntv2": {k: en[k] for k in ('input_ids', 'attention_mask')},
        "prokbert": {k: ep[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')},
        "grammar": {"onehot": oh},
    }

@torch.no_grad()
def forward_probs(model, data, batch=192, use_amp=True):
    n = data['ntv2']['input_ids'].shape[0]
    out = []
    for i in range(0, n, batch):
        sub = {v: ({k: t[i:i + batch] for k, t in d.items()}) for v, d in data.items()}
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logit, _ = model(sub)
        out.append(torch.softmax(logit.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def find_ckpt(species, fold=1):
    files = glob.glob(os.path.join(RUN_DIR, f's{species}', str(fold), 'checkpoints', '*.pth'))
    return files[0] if files else None

def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=192)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    n_target = 20 if args.smoke else args.n

    out_p = os.path.join(_HERE, 'ism_v2.npz')
    if os.path.exists(out_p) and not args.force and not args.smoke:
        print(f'exists {out_p} (--force to overwrite), skipped')
        return

    df = pd.read_csv(os.path.join(DATA_DIR, 'Test', f'{SPECIES}_test.csv'))
    pos = df[df.label == 1]['text'].tolist()
    pos = [s for s in pos if len(s) == L and set(s) <= set(BASES)]
    rng = np.random.default_rng(args.seed)
    if len(pos) > n_target:
        sel = np.sort(rng.choice(len(pos), n_target, replace=False))
        seqs = [pos[i] for i in sel]
    else:
        sel = np.arange(len(pos)); seqs = pos
    n_seq = len(seqs)
    print(f's{SPECIES} (E. coli): {len(pos)} positives, sampling {n_seq} for ISM')

    TOK = build_tokenizers_v2()
    cfg = DeepProV2Config()
    model = DeepProV2(cfg, n_train=1, use_moe=True, gating="static",
                      use_grammar=True, router="softmax", gate_temperature=1.0).to(device).eval()
    ckpt = find_ckpt(SPECIES, 1)
    assert ckpt is not None, f'no fold-1 checkpoint: s{SPECIES}'
    sd = torch.load(ckpt, map_location='cpu'); sd.pop(EMB_KEY, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f'unexpected: {unexpected[:5]}'
    assert set(missing) <= {EMB_KEY}, f'missing: {set(missing) - {EMB_KEY}}'
    model.eval()
    print(f'loading ckpt: {os.path.basename(ckpt)}')

    imp_sum = np.zeros(L)
    heat_sum = np.zeros((4, L))
    p_wt = np.zeros(n_seq)
    t0 = time.time()
    for s_i, src in enumerate(seqs):
        items = [src]; m_pos, m_base = [], []
        for i, ch in enumerate(src):
            for b in BASES:
                if b == ch:
                    continue
                items.append(src[:i] + b + src[i + 1:])
                m_pos.append(i); m_base.append(BASES.index(b))
        probs = forward_probs(model, tokenize(items), batch=args.batch)
        p0 = probs[0]; p_wt[s_i] = p0
        dp = probs[1:] - p0
        m_pos = np.array(m_pos); m_base = np.array(m_base)
        np.add.at(heat_sum, (m_base, m_pos), dp)
        np.add.at(imp_sum, m_pos, -dp)
        if (s_i + 1) % 25 == 0 or s_i == n_seq - 1:
            el = time.time() - t0
            print(f'  [{s_i+1}/{n_seq}] mean p_wt={p_wt[:s_i+1].mean():.3f} '
                  f'{el:.1f}s (~{el/(s_i+1):.2f}s/seq)', flush=True)

    importance = imp_sum / (n_seq * 3.0)
    heat = heat_sum / n_seq
    np.savez_compressed(out_p, importance=importance.astype(np.float32),
                        heat=heat.astype(np.float32), p_wt=p_wt.astype(np.float32),
                        idx=sel.astype(np.int64), bases=np.array(list(BASES)),
                        tss=TSS, n_seq=n_seq)
    print(f'\nsaved -> {out_p}')
    peaks = [int(i - TSS if i < TSS else i - TSS + 1) for i in np.argsort(importance)[::-1][:6]]
    print(f'importance top-6 peaks (rel TSS): {peaks}')

if __name__ == '__main__':
    main()
