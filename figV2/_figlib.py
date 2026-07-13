import os
import re
import glob
import sys

import numpy as np
from scipy.special import softmax

import importlib.util as _ilu
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGV1 = os.path.join(os.path.dirname(_HERE), 'figV1')
_spec = _ilu.spec_from_file_location('_figlib_v1', os.path.join(_FIGV1, '_figlib.py'))
_v1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
set_style, SPECIES_NAME, save_panels = _v1.set_style, _v1.SPECIES_NAME, _v1.save_panels

PROJ_ROOT = os.path.dirname(_HERE)

RUNS = {
    'deeppro_v2':    'iProV2/models_log/DeepPro-v2/2026-07-13_11-48',
    'msBERT':        'baselines/msBERT/models_log/msBERT/2026-06-10_22-25',
    'DNABERT2-CAMP': 'baselines/dnabert2_camp/models_log/DNABERT2_CAMP/2026-06-08_23-12',
    'GROVER':        'baselines/grover/models_log/GROVER/2026-07-11_17-49',
    'HyenaDNA':      'baselines/hyenadna/models_log/HyenaDNA/2026-07-12_04-57',
}

COLORS = {
    'deeppro_v2':    '#C1272D',
    'msBERT':        '#0072B2',
    'DNABERT2-CAMP': '#56B4E9',
    'GROVER':        '#009E73',
    'HyenaDNA':      '#7F7F7F',
}
DISPLAY = {
    'deeppro_v2': 'DeepPro-v2', 'msBERT': 'msBERT', 'DNABERT2-CAMP': 'DNABERT2-CAMP',
    'GROVER': 'GROVER', 'HyenaDNA': 'HyenaDNA',
}
MARKERS = {
    'deeppro_v2': 'o', 'msBERT': 's', 'DNABERT2-CAMP': 'D', 'GROVER': '^', 'HyenaDNA': 'v',
}
BASELINES = ['msBERT', 'DNABERT2-CAMP', 'GROVER', 'HyenaDNA']
HERO = 'deeppro_v2'

VIEW_COLORS = ['#4C72B0', '#5B8FC9', '#9C6BB0', '#55A868', '#C9A227']
VIEW_NAMES = ['DNABERT 3-mer', 'DNABERT 6-mer', 'DNABERT-2', 'NT-v2', 'ProkBERT-mini-c']
VIEW_SHORT = ['DB-3', 'DB-6', 'DB-2', 'NT-v2', 'Prok']
FUSION_COLOR = '#C1272D'

REPR_CACHE_V2 = os.path.join(PROJ_ROOT, 'iProV2', 'repr_cache_v2')
CROSS_SPECIES_V2 = os.path.join(PROJ_ROOT, 'iProV2', 'cross_species')

def _parse_epoch(fname):
    m = re.search(r'_epoch(\d+)_', os.path.basename(fname))
    return int(m.group(1)) if m else 1

def _fold_logits(fold_dir, mode='test'):
    ck = glob.glob(os.path.join(fold_dir, 'checkpoints', '*.pth')) + \
         glob.glob(os.path.join(fold_dir, 'checkpoints', '*.txt'))
    epoch = _parse_epoch(ck[0]) if ck else 1
    yt = np.load(os.path.join(fold_dir, 'draw_data', f'{mode}_y_true.npy'))
    sc = np.load(os.path.join(fold_dir, 'draw_data', f'{mode}_score.npy'))
    idx = epoch - 1 if sc.shape[0] > 1 else 0
    return yt[idx], sc[idx]

SCORE_IS_PROB = {'msBERT'}

def load_species_ensemble(method, sid, mode='test', proj_root=None):
    proj_root = proj_root or PROJ_ROOT
    base = os.path.join(proj_root, RUNS[method])
    sp_dir = os.path.join(base, f's{sid}')
    folds, y_true = [], None
    for f in range(1, 6):
        fd = os.path.join(sp_dir, str(f))
        if not os.path.isdir(fd):
            continue
        yt, lg = _fold_logits(fd, mode)
        y_true = yt
        folds.append(lg)
    if not folds:
        return None, None
    if method in SCORE_IS_PROB:
        prob = np.mean(np.stack([f[:, 1] for f in folds], 0), 0)
    else:
        avg = np.mean(np.stack(folds, 0), 0)
        prob = softmax(avg, axis=1)[:, 1]
    return y_true.astype(int), prob

def load_species_oof(method, sid, proj_root=None):
    proj_root = proj_root or PROJ_ROOT
    base = os.path.join(proj_root, RUNS[method])
    sp_dir = os.path.join(base, f's{sid}')
    yts, prs = [], []
    for f in range(1, 6):
        fd = os.path.join(sp_dir, str(f))
        if not os.path.isdir(fd):
            continue
        yt, sc = _fold_logits(fd, 'val')
        pr = sc[:, 1] if method in SCORE_IS_PROB else softmax(sc, axis=1)[:, 1]
        prs.append(pr)
        yts.append(yt.astype(int))
    if not yts:
        return None, None
    return np.concatenate(yts), np.concatenate(prs)

def load_repr_cache(sid):
    p = os.path.join(REPR_CACHE_V2, f's{sid}.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    out = {f'feat{k}': d[f'feat{k}'].astype(np.float32) for k in range(5)}
    out['rep'] = d['rep'].astype(np.float32)
    out['logits'] = d['logits'].astype(np.float32)
    out['gate'] = d['gate'].astype(np.float32)
    out['y'] = d['y'].astype(int)
    return out

ALL_SPECIES = list(range(1, 24))
