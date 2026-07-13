from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from data.physics import one_hot_batch, dg_profile, melting_target_batch, BASES
from models.melting_head import MeltingHead, MeltingHeadConfig

DATA_ROOT = r"../data/Benchmark Dataset/csv"
SEED = 42

def load_split(species: int):
    base = f"{DATA_ROOT}/Train_5fold/{species}"
    tr = pd.read_csv(f"{base}/fold_1_train.csv")
    va = pd.read_csv(f"{base}/fold_1_val.csv")
    te = pd.read_csv(f"{DATA_ROOT}/Test/{species}_test.csv")
    return tr, va, te

def dg_profile_batch(seqs, length: int = 81) -> np.ndarray:
    return np.stack([dg_profile(s, length) for s in seqs], axis=0)

def gc_content(seqs, length: int = 81) -> np.ndarray:
    oh = one_hot_batch(seqs, length)
    g_idx, c_idx = BASES.index("G"), BASES.index("C")
    gc = oh[:, [g_idx, c_idx], :].sum(axis=(1, 2)) / length
    return gc.reshape(-1, 1).astype(np.float32)

def lr_probe(Xtr, ytr, Xte, yte):
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, prob)
    except ValueError:
        auc = float("nan")
    return matthews_corrcoef(yte, pred), auc

class MeltProbe(nn.Module):

    def __init__(self):
        super().__init__()
        self.melt = MeltingHead(MeltingHeadConfig())
        self.clf = nn.Sequential(nn.Linear(self.melt.cfg.dh, 32), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, one_hot):
        me = self.melt(one_hot)
        return self.clf(me["melt"]), me

def evaluate(model, oh, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(oh)
        prob = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        pred = logits.argmax(-1).cpu().numpy()
    yt = y.cpu().numpy()
    try:
        auc = roc_auc_score(yt, prob)
    except ValueError:
        auc = float("nan")
    return matthews_corrcoef(yt, pred), auc

def train_melt_e2e(tr, va, te, device, epochs, lr, batch, lambda_melt):
    def tens(df):
        seqs = df["text"].tolist()
        oh = torch.from_numpy(one_hot_batch(seqs)).to(device)
        tgt = torch.from_numpy(melting_target_batch(seqs)).to(device)
        y = torch.tensor(df["label"].values, dtype=torch.long, device=device)
        return oh, tgt, y

    oh_tr, tgt_tr, y_tr = tens(tr)
    oh_va, _, y_va = tens(va)
    oh_te, _, y_te = tens(te)

    torch.manual_seed(SEED)
    model = MeltProbe().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = len(y_tr)

    best = {"val": -1.0, "test_mcc": float("nan"), "test_auc": float("nan"), "ep": -1}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits, me = model(oh_tr[idx])
            loss = ce(logits, y_tr[idx])
            if lambda_melt > 0:
                warm = max(0.0, 1.0 - (ep - 1) / 10.0)
                loss = loss + warm * lambda_melt * MeltingHead.melt_loss(me["rho"], tgt_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v_mcc, _ = evaluate(model, oh_va, y_va)
        if v_mcc > best["val"]:
            t_mcc, t_auc = evaluate(model, oh_te, y_te)
            best.update(val=v_mcc, test_mcc=t_mcc, test_auc=t_auc, ep=ep)

    model.eval()
    with torch.no_grad():
        _, me = model(oh_te)
        rho = me["rho"].cpu().numpy()
    pos = y_te.cpu().numpy() == 1
    rho_pos = rho[pos].mean(axis=0)
    m10 = rho_pos[49:57].mean()
    rest = np.concatenate([rho_pos[:49], rho_pos[57:]]).mean()
    return best, (m10, rest, int(np.argmax(rho_pos)))

def run_species(sp, device, epochs, lr, batch, lambda_melt):
    tr, va, te = load_split(sp)
    seqs_tr, seqs_te = tr["text"].tolist(), te["text"].tolist()
    ytr, yte = tr["label"].values, te["label"].values

    row = {"species": sp, "n_tr": len(tr), "n_te": len(te), "pos%": float(tr["label"].mean())}

    row["majority"] = 0.0

    row["gc"], _ = lr_probe(gc_content(seqs_tr), ytr, gc_content(seqs_te), yte)

    row["dg81"], _ = lr_probe(dg_profile_batch(seqs_tr), ytr, dg_profile_batch(seqs_te), yte)

    row["melt81"], _ = lr_probe(melting_target_batch(seqs_tr), ytr,
                                melting_target_batch(seqs_te), yte)

    row["onehot"], _ = lr_probe(one_hot_batch(seqs_tr).reshape(len(tr), -1), ytr,
                                one_hot_batch(seqs_te).reshape(len(te), -1), yte)

    best, rho_info = train_melt_e2e(tr, va, te, device, epochs, lr, batch, lambda_melt)
    row["melthead_e2e"] = best["test_mcc"]
    row["_rho"] = rho_info
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", type=int, nargs="+", default=[10, 2, 15, 20])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lambda_melt", type=float, default=0.3)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe_melt] species={args.species} device={device} seed={SEED} "
          f"lambda_melt={args.lambda_melt}\n")

    cols = ["majority", "gc", "dg81", "melt81", "onehot", "melthead_e2e"]
    header = (f"{'sp':>3} {'n_te':>6} {'pos%':>5} | "
              + " ".join(f"{c:>12}" for c in cols))
    print(header)
    print("-" * len(header))

    rows = []
    for sp in args.species:
        t0 = time.time()
        r = run_species(sp, device, args.epochs, args.lr, args.batch, args.lambda_melt)
        rows.append(r)
        line = (f"{r['species']:>3} {r['n_te']:>6} {r['pos%']:>5.2f} | "
                + " ".join(f"{r[c]:>12.4f}" for c in cols))
        m10, rest, peak = r["_rho"]
        print(line + f"   [ρ −10={m10:.3f} rest={rest:.3f} peak@{peak}]  ({time.time()-t0:.1f}s)")

    print("-" * len(header))
    mean = {c: float(np.mean([r[c] for r in rows])) for c in cols}
    print(f"{'avg':>3} {'':>6} {'':>5} | " + " ".join(f"{mean[c]:>12.4f}" for c in cols))

    print("\n" + "=" * 66)
    dg, melt, gc, oh, e2e = mean["dg81"], mean["melt81"], mean["gc"], mean["onehot"], mean["melthead_e2e"]
    phys = max(dg, melt)
    print(f"physics-only best (dg81/melt81)={phys:.4f} | GC scalar={gc:.4f} | one-hot ceiling={oh:.4f} | melting-head e2e={e2e:.4f}")
    if phys < 0.10:
        print("[verdict] physics-only ~ chance -> melting physics has almost no independent signal.")
    elif phys <= gc + 0.03:
        print("[verdict] physics-only ~ GC scalar -> melting signal is basically base composition, no positional gain.")
    else:
        print("[verdict] physics-only > GC scalar -> melting physics has genuine positional signal (beyond base composition).")
    if e2e > phys + 0.05:
        print("       melting-head e2e >> physics-only -> the gain comes from a generic CNN, not melting itself.")
    print("=" * 66)

if __name__ == "__main__":
    main()
