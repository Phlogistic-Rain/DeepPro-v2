from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn

_IPROV2 = os.path.abspath(os.getcwd())
_ARCHIVE = os.path.join(_IPROV2, "_archive_v2_scratch")
for _p in (_ARCHIVE, _IPROV2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_t0 import load_split, to_tensors, MechanismProbe, SEED
from probe_interpret import pcc_to_consensus, consensus_str, CONS10
from models.melting_head import MeltingHead
from data.physics import one_hot_batch, BASES

OUT = os.path.abspath(os.path.join(os.getcwd(), "..", "figV2", "05.motif_grammar", "shuffle_data.npz"))
SPECIES, EPOCHS, LR, BATCH, W10 = 10, 60, 1e-3, 512, 6

def train_shuffled(species, epochs, lr, batch, device, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = load_split(species)
    oh_tr, tgt_tr, y_tr = to_tensors(tr, device)
    oh_te, _, y_te = to_tensors(te, device)
    g = torch.Generator(device=device).manual_seed(seed)
    y_tr = y_tr[torch.randperm(len(y_tr), generator=g, device=device)]
    model = MechanismProbe().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = len(y_tr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits, mo, me = model(oh_tr[idx])
            warm = max(0.0, 1.0 - (ep - 1) / 10.0)
            loss = (ce(logits, y_tr[idx])
                    + 0.05 * model.moe.pwm_sharpness_loss()
                    + warm * (0.3 * MeltingHead.melt_loss(me["rho"], tgt_tr[idx])
                              + 0.1 * model.moe.load_balance_loss(mo["g"])))
            opt.zero_grad(); loss.backward(); opt.step()
    return model, oh_te, y_te

def data_derived_minus10(seqs, w=W10):
    oh = one_hot_batch(seqs)
    pfm = oh.mean(axis=0)
    eps = 1e-9
    ic_pos = 2.0 + (pfm * np.log2(pfm + eps)).sum(axis=0)
    L = pfm.shape[1]
    best_s, best_ic = 0, -1.0
    for s in range(0, L - w + 1):
        m = ic_pos[s:s + w].mean()
        if m > best_ic:
            best_ic, best_s = m, s
    win = pfm[:, best_s:best_s + w].T
    return win, best_s, ic_pos

N_SEED = 5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shuffle] species={SPECIES} epochs={EPOCHS} device={device} seeds={N_SEED}")

    P10_seed0 = None
    null_best_r = []
    r10_seed0 = None
    for si in range(N_SEED):
        model, _, _ = train_shuffled(SPECIES, EPOCHS, LR, BATCH, device, seed=SEED + si)
        model.eval()
        with torch.no_grad():
            pw = model.moe.pwm_probs()
        P10s = pw["P10"].cpu().numpy()
        K = P10s.shape[0]
        r10s = np.array([pcc_to_consensus(P10s[e].T, CONS10) for e in range(K)])
        null_best_r.append(float(r10s.max()))
        if si == 0:
            P10_seed0, r10_seed0 = P10s, r10s
        print(f"  seed{SEED+si}: best −10 r={r10s.max():+.2f} "
              f"({consensus_str(P10s[int(np.argmax(r10s))].T)}) mean={r10s.mean():+.2f}")
    null_best_r = np.array(null_best_r)
    best10s = int(np.argmax(r10_seed0))
    print(f"[shuffle-null] best-expert r10: mean={null_best_r.mean():+.3f} "
          f"max={null_best_r.max():+.3f} (real deployed best = 0.67)")

    tr, va, te = load_split(SPECIES)
    pos_seqs = te.loc[te["label"] == 1, "text"].tolist()
    dd_win, dd_start, ic_pos = data_derived_minus10(pos_seqs)
    r_dd = pcc_to_consensus(dd_win, CONS10)
    print(f"[data-derived] −10 window start={dd_start} consensus={consensus_str(dd_win)} "
          f"r_vs_TATAAT={r_dd:+.2f}")

    np.savez(OUT, P10_shuffle=P10_seed0, r10_shuffle=r10_seed0, best10_shuffle=best10s,
             null_best_r=null_best_r, dd_win=dd_win, dd_start=dd_start,
             dd_consensus=consensus_str(dd_win), r_dd=r_dd, ic_pos=ic_pos, cons10=CONS10)
    print(f"[saved] {OUT}")

if __name__ == "__main__":
    main()
