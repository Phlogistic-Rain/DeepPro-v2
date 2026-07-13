import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

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
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

N_SPECIES = 23
DATA_DIR = os.path.join(_ROOT, 'data', 'Benchmark Dataset', 'csv')
RUN_DIR = os.path.join(_INNER, 'models_log', 'DeepPro-v2', '2026-07-13_11-48')
OUT_DIR = os.path.join(_INNER, 'cross_species')
EMB_KEY = 'FusionNet.emb.weight'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    cfg = DeepProV2Config()
    model = DeepProV2(cfg, n_train=1, use_moe=True, gating="static",
                      use_grammar=True, router="softmax", gate_temperature=1.0)
    return model.to(device).eval()

def find_ckpt(species: int, fold: int):
    files = glob.glob(os.path.join(RUN_DIR, f's{species}', str(fold), 'checkpoints', '*.pth'))
    return files[0] if files else None

@torch.no_grad()
def infer_logits(model, data, onehot, n, batch=384, use_amp=True):
    outs = []
    for i in range(0, n, batch):
        sub = {v: {k: t[i:i + batch] for k, t in d.items()} for v, d in data.items()}
        sub['grammar'] = {'onehot': onehot[i:i + batch]}
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logit, _rep = model(sub)
        outs.append(logit.float().cpu().numpy())
    return np.concatenate(outs, axis=0)

def load_test_tensors():
    tests = {}
    for j in range(1, N_SPECIES + 1):
        ds = TextDatasetV2(os.path.join(DATA_DIR, 'Test', f'{j}_test.csv'))
        tests[j] = {'data': ds.tokenized, 'onehot': ds.onehot,
                    'y': ds.labels.numpy().astype(np.int64), 'n': len(ds)}
    return tests

def load_fold(model, src, fold):
    ckpt = find_ckpt(src, fold)
    if ckpt is None:
        return False
    sd = torch.load(ckpt, map_location='cpu')
    sd.pop(EMB_KEY, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    assert set(missing) <= {EMB_KEY}, f'unexpected missing: {set(missing) - {EMB_KEY}}'
    model.eval()
    return True

def run_source(model, src, tests, folds=range(1, 6), batch=384):
    accum = {j: None for j in range(1, N_SPECIES + 1)}
    nfolds = 0
    for f in folds:
        if not load_fold(model, src, f):
            print(f'  [warn] s{src}/fold{f}: no checkpoint, skipped')
            continue
        nfolds += 1
        for j in range(1, N_SPECIES + 1):
            lg = infer_logits(model, tests[j]['data'], tests[j]['onehot'], tests[j]['n'], batch)
            accum[j] = lg if accum[j] is None else accum[j] + lg
    if nfolds == 0:
        return None
    out = {'nfolds': np.array(nfolds)}
    for j in range(1, N_SPECIES + 1):
        out[f'logits_{j}'] = accum[j] / nfolds
        out[f'ytrue_{j}'] = tests[j]['y']
    return out

def assemble():
    mcc = np.full((N_SPECIES, N_SPECIES), np.nan)
    auc = np.full((N_SPECIES, N_SPECIES), np.nan)
    acc = np.full((N_SPECIES, N_SPECIES), np.nan)
    missing_src = []
    for i in range(1, N_SPECIES + 1):
        p = os.path.join(OUT_DIR, f'src{i}.npz')
        if not os.path.exists(p):
            missing_src.append(i)
            continue
        d = np.load(p)
        for j in range(1, N_SPECIES + 1):
            lg = d[f'logits_{j}']; yt = d[f'ytrue_{j}']
            prob = softmax(lg, axis=1)[:, 1]
            yp = (prob >= 0.5).astype(np.int64)
            mcc[i - 1, j - 1] = matthews_corrcoef(yt, yp)
            acc[i - 1, j - 1] = accuracy_score(yt, yp)
            auc[i - 1, j - 1] = roc_auc_score(yt, prob)
    np.save(os.path.join(OUT_DIR, 'mcc_matrix.npy'), mcc)
    np.save(os.path.join(OUT_DIR, 'auc_matrix.npy'), auc)
    np.save(os.path.join(OUT_DIR, 'acc_matrix.npy'), acc)
    if missing_src:
        print(f'[assemble] warning: missing source species {missing_src} (rows are NaN)')
    diag = np.array([mcc[k, k] for k in range(N_SPECIES)])
    print(f'[assemble] diagonal MCC mean={np.nanmean(diag):.4f}')
    print(f'[assemble] off-diag MCC mean={np.nanmean(mcc[~np.eye(N_SPECIES, dtype=bool)]):.4f}')
    print(f'[assemble] matrices saved: {OUT_DIR}/(mcc|auc|acc)_matrix.npy')
    return mcc, auc, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=384)
    ap.add_argument('--sources', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--assemble-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.assemble_only:
        assemble()
        return

    t0 = time.time()
    print('tokenizing 23 test sets ...')
    tests = load_test_tensors()
    print(f'  test sets ready, elapsed {time.time() - t0:.1f}s')

    model = build_model()

    if args.smoke:
        t1 = time.time()
        out = run_source(model, 1, tests, folds=range(1, 2), batch=args.batch)
        dt = time.time() - t1
        lg = out['logits_1']; yt = out['ytrue_1']
        yp = (softmax(lg, 1)[:, 1] >= 0.5).astype(int)
        print(f'[smoke] s1 fold1 -> 23 targets elapsed {dt:.1f}s  '
              f'full ETA ~ {dt * 5 * 23 / 60:.0f} min')
        print(f'[smoke] s1->s1(fold1) MCC={matthews_corrcoef(yt, yp):.4f} '
              f'AUC={roc_auc_score(yt, softmax(lg,1)[:,1]):.4f}')
        return

    sources = [int(x) for x in args.sources.split(',')] if args.sources else list(range(1, N_SPECIES + 1))
    for i in sources:
        out_p = os.path.join(OUT_DIR, f'src{i}.npz')
        if os.path.exists(out_p) and not args.force:
            print(f's{i}: exists, skipped')
            continue
        ts = time.time()
        print(f's{i}: inferring ...', flush=True)
        out = run_source(model, i, tests, batch=args.batch)
        if out is None:
            print(f's{i}: no valid fold, skipped')
            continue
        np.savez_compressed(out_p, **out)
        print(f's{i}: done ({int(out["nfolds"])} folds), elapsed {time.time() - ts:.1f}s -> {out_p}', flush=True)

    assemble()

if __name__ == '__main__':
    main()
