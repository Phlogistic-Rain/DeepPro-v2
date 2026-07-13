import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast

_INNER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_INNER)
for _p in (_ROOT, _INNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iProV2.dataset import TextDatasetV2
from iProV2.models.deeppro_v2 import DeepProV2, DeepProV2Config

import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

N_SPECIES = 23
DATA_DIR = os.path.join(_ROOT, 'data', 'Benchmark Dataset', 'csv')
RUN_DIR = os.path.join(_INNER, 'models_log', 'DeepPro-v2', '2026-07-13_11-48')
OUT_DIR = os.path.join(_INNER, 'repr_cache_v2')
EMB_KEY = 'FusionNet.emb.weight'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    cfg = DeepProV2Config()
    model = DeepProV2(cfg, n_train=1, use_moe=True, gating="static",
                      use_grammar=True, router="softmax", gate_temperature=1.0)
    return model.to(device).eval()

def load_fold(model, species, fold=1):
    files = glob.glob(os.path.join(RUN_DIR, f's{species}', str(fold), 'checkpoints', '*.pth'))
    if not files:
        return False
    sd = torch.load(files[0], map_location='cpu')
    sd.pop(EMB_KEY, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    assert set(missing) <= {EMB_KEY}, f'unexpected missing: {set(missing) - {EMB_KEY}}'
    model.eval()
    return True

@torch.no_grad()
def infer_species(model, ds, batch=256, use_amp=True):
    n = len(ds)
    data = ds.tokenized
    onehot_all = ds.onehot
    fb = {k: [] for k in range(5)}
    reps, logs = [], []
    gate = None
    for i in range(0, n, batch):
        sub = {v: {k: t[i:i + batch] for k, t in d.items()} for v, d in data.items()}
        oh = onehot_all[i:i + batch].to(device, non_blocking=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            feats = model.extract_features(sub)
            rep, g = model.FusionNet.fuse(feats, None)
            m = model.grammar(oh)["m"]
            logit = model.cls_head(torch.cat([rep, m], dim=-1))
        for k in range(5):
            fb[k].append(feats[k].float().cpu().numpy())
        reps.append(rep.float().cpu().numpy())
        logs.append(logit.float().cpu().numpy())
        gate = g.float().cpu().numpy()
    feats_np = {k: np.concatenate(fb[k], 0) for k in range(5)}
    return feats_np, np.concatenate(reps, 0), np.concatenate(logs, 0), gate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    model = build_model()
    species = [10] if args.smoke else list(range(1, N_SPECIES + 1))

    for i in species:
        out_p = os.path.join(OUT_DIR, f's{i}.npz')
        if os.path.exists(out_p) and not args.force:
            print(f's{i}: exists, skipped')
            continue
        if not load_fold(model, i, 1):
            print(f's{i}: no fold-1 checkpoint, skipped')
            continue

        ds = TextDatasetV2(os.path.join(DATA_DIR, 'Test', f'{i}_test.csv'))
        y = ds.labels.numpy().astype(np.int64)
        ts = time.time()
        feats, rep, logits, gate = infer_species(model, ds, args.batch)
        save = {f'feat{k}': feats[k].astype(np.float16) for k in range(5)}
        save['rep'] = rep.astype(np.float16)
        save['logits'] = logits.astype(np.float32)
        save['gate'] = gate.astype(np.float32)
        save['y'] = y
        np.savez_compressed(out_p, **save)
        dims = {k: feats[k].shape[1] for k in range(5)}
        print(f's{i}: N={len(ds)} rep={rep.shape[1]} viewdims={dims} '
              f'pos={int(y.sum())} gate={np.round(gate,3).tolist()} '
              f'elapsed {time.time()-ts:.1f}s → {out_p}')
        if args.smoke:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            auc = cross_val_score(LogisticRegression(max_iter=500),
                                  rep.astype(np.float32), y, cv=3, scoring='roc_auc').mean()
            print(f'[smoke] s10 rep 3-fold CV AUC = {auc:.4f}')

if __name__ == '__main__':
    main()
