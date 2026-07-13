import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.amp import autocast
from scipy.special import softmax
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

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
EMB_KEY = "FusionNet.emb.weight"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model() -> DeepProV2:
    cfg = DeepProV2Config()
    model = DeepProV2(cfg, n_train=1, use_moe=True, gating="static",
                      use_grammar=True, router="softmax", gate_temperature=1.0)
    return model.to(device).eval()

def find_ckpts(weights_dir: str, species: int):
    pats = [
        os.path.join(weights_dir, f"s{species}", "*", "checkpoints", "*.pth"),
        os.path.join(weights_dir, f"s{species}", "*", "*.pth"),
        os.path.join(weights_dir, f"s{species}", "*.pth"),
    ]
    seen, out = set(), []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            rf = os.path.realpath(f)
            if rf not in seen:
                seen.add(rf)
                out.append(f)
    return out

def load_ckpt(model: DeepProV2, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    sd.pop(EMB_KEY, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in {ckpt_path}: {unexpected[:5]}"
    assert set(missing) <= {EMB_KEY}, f"unexpected missing keys: {set(missing) - {EMB_KEY}}"
    model.eval()

@torch.no_grad()
def infer_logits(model, tokenized, onehot, n, batch=384, use_amp=True):
    outs = []
    for i in range(0, n, batch):
        sub = {v: {k: t[i:i + batch] for k, t in d.items()} for v, d in tokenized.items()}
        sub["grammar"] = {"onehot": onehot[i:i + batch]}
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
            logit, _rep = model(sub)
        outs.append(logit.float().cpu().numpy())
    return np.concatenate(outs, axis=0)

def metrics_from(y_true, prob):
    y_pred = (prob >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Sn": recall_score(y_true, y_pred, zero_division=0),
        "Sp": sp,
        "Pre": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, prob),
    }

def run_species(model, species, data_dir, weights_dir, out_dir, batch, use_amp):
    test_csv = os.path.join(data_dir, "Test", f"{species}_test.csv")
    if not os.path.exists(test_csv):
        print(f"[s{species}] missing test data: {test_csv} -- skipped (see root README: download data from Zenodo)")
        return None

    ckpts = find_ckpts(weights_dir, species)
    if not ckpts:
        print(f"[s{species}] no weights found: {os.path.join(weights_dir, f's{species}')}/... "
              f"-- skipped (see root README: download weights from Zenodo)")
        return None

    ds = TextDatasetV2(test_csv)
    y_true = ds.labels.numpy().astype(np.int64)
    n = len(ds)

    accum = None
    for cp in ckpts:
        load_ckpt(model, cp)
        lg = infer_logits(model, ds.tokenized, ds.onehot, n, batch=batch, use_amp=use_amp)
        accum = lg if accum is None else accum + lg
    assert accum is not None
    logits = accum / len(ckpts)
    prob = softmax(logits, axis=1)[:, 1]
    m = metrics_from(y_true, prob)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"s{species}_pred.npz"),
            logits=logits, prob=prob, y_true=y_true, n_folds=np.array(len(ckpts)),
        )
    print(f"[s{species}] n={n:5d}  folds={len(ckpts)}  "
          f"MCC={m['MCC']:.4f}  ACC={m['ACC']:.4f}  AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")
    return m

def parse_species(v: str):
    v = v.strip().lower()
    if v == "all":
        return list(range(1, N_SPECIES + 1))
    return [int(x) for x in v.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="DeepPro-v2 minimal inference on benchmark test set")
    ap.add_argument("--species", type=str, default="1",
                    help='species: int / comma list (e.g. "1,10,17") / "all" (1..23)')
    ap.add_argument("--data_dir", type=str, default=os.path.join(_ROOT, "data", "Benchmark Dataset", "csv"),
                    help="benchmark csv root (has Test/{sp}_test.csv); default <root>/data/Benchmark Dataset/csv")
    ap.add_argument("--weights_dir", type=str, default=os.path.join(_ROOT, "weights"),
                    help="weights root (has s{sp}/{fold}/checkpoints/*.pth); default <root>/weights")
    ap.add_argument("--out_dir", type=str, default=os.path.join(_INNER, "infer_out"),
                    help="output dir for predictions (s{sp}_pred.npz); empty string disables saving")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (on by default)")
    args = ap.parse_args()

    species_list = parse_species(args.species)
    print(f"device={device}  species={species_list}")
    print(f"data_dir   = {args.data_dir}")
    print(f"weights_dir= {args.weights_dir}\n")

    model = build_model()

    results = {}
    for sp in species_list:
        m = run_species(model, sp, args.data_dir, args.weights_dir,
                        args.out_dir, args.batch, use_amp=not args.no_amp)
        if m is not None:
            results[sp] = m

    if results:
        keys = ["ACC", "Sn", "Sp", "Pre", "F1", "MCC", "AUC"]
        mean = {k: float(np.mean([results[s][k] for s in results])) for k in keys}
        print("\n===== mean over {} species =====".format(len(results)))
        print("  " + "  ".join(f"{k}={mean[k]:.4f}" for k in keys))

if __name__ == "__main__":
    main()
