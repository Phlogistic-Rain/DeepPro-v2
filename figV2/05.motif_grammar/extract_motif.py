from __future__ import annotations

import os
import sys

import numpy as np
import torch

_IPROV2 = os.path.abspath(os.getcwd())
_ARCHIVE = os.path.join(_IPROV2, "_archive_v2_scratch")
for _p in (_ARCHIVE, _IPROV2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_interpret import train_branch, consensus_str, pcc_to_consensus, CONS35, CONS10

OUT = os.path.abspath(os.path.join(os.getcwd(), "..", "figV2", "05.motif_grammar", "motif_data.npz"))
SPECIES, EPOCHS, LR, BATCH = 10, 60, 1e-3, 512

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] species={SPECIES} (E. coli) epochs={EPOCHS} device={device}")

    model, oh_te, y_te = train_branch(SPECIES, EPOCHS, LR, BATCH, device)
    model.eval()
    with torch.no_grad():
        out = model.moe(oh_te)
        pw = model.moe.pwm_probs()
    P35 = pw["P35"].cpu().numpy()
    P10 = pw["P10"].cpu().numpy()
    log_phi = pw["log_phi"].cpu().numpy()
    g_all = out["g"].cpu().numpy()

    pos_mask = (y_te.cpu().numpy() == 1)
    usage = g_all[pos_mask].mean(axis=0)
    K = P35.shape[0]

    c35, c10, r35, r10 = [], [], [], []
    for e in range(K):
        c35.append(consensus_str(P35[e].T)); c10.append(consensus_str(P10[e].T))
        r35.append(pcc_to_consensus(P35[e].T, CONS35))
        r10.append(pcc_to_consensus(P10[e].T, CONS10))
    r35, r10 = np.array(r35), np.array(r10)

    print(f"\nexpert | -35 (vs {CONS35}) | -10 (vs {CONS10}) | usage%")
    for e in range(K):
        print(f"  E{e}: {c35[e]} (r={r35[e]:+.2f}) | {c10[e]} (r={r10[e]:+.2f}) | {usage[e]*100:5.1f}%")
    best10, best35 = int(np.argmax(r10)), int(np.argmax(r35))
    print(f"\nbest -10 match: E{best10} ({c10[best10]}, r={r10[best10]:+.2f})  "
          f"best -35 match: E{best35} ({c35[best35]}, r={r35[best35]:+.2f})")

    d_grid = np.arange(model.moe.cfg.d_min, model.moe.cfg.d_max + 1)
    np.savez(
        OUT,
        P35=P35, P10=P10, log_phi=log_phi, usage=usage, d_grid=d_grid,
        c35=np.array(c35), c10=np.array(c10), r35=r35, r10=r10,
        best10=best10, best35=best35, cons35=CONS35, cons10=CONS10,
        species=SPECIES,
    )
    print(f"\n[saved] {OUT}")

if __name__ == "__main__":
    main()
