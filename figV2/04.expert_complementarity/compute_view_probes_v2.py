import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _figlib as F

SIDS = F.ALL_SPECIES
NV = 5

def oof_proba(X, y):
    X = StandardScaler().fit_transform(X.astype(np.float32))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')
    return cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]

def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Yc.T @ Xc) ** 2
    xx = np.linalg.norm(Xc.T @ Xc)
    yy = np.linalg.norm(Yc.T @ Yc)
    return float(xy / (xx * yy + 1e-12))

def main():
    A_v = np.zeros((23, NV)); A_f = np.zeros(23)
    Acc_v = np.zeros((23, NV)); Acc_f = np.zeros(23)
    Acc_best = np.zeros(23); Acc_oracle = np.zeros(23)
    Dis = np.zeros((23, NV, NV))
    Uni = np.zeros((23, NV))
    CKA = np.zeros((23, NV, NV))
    N = np.zeros(23, int); P = np.zeros(23, int)

    for r, sid in enumerate(SIDS):
        c = F.load_repr_cache(sid)
        y = c['y'].astype(int)
        N[r] = len(y); P[r] = int(y.sum())
        feats = [c[f'feat{v}'].astype(np.float32) for v in range(NV)]
        preds = []
        for v in range(NV):
            p = oof_proba(feats[v], y)
            preds.append((p >= 0.5).astype(int))
            A_v[r, v] = roc_auc_score(y, p)
            Acc_v[r, v] = (preds[v] == y).mean()
        pf = oof_proba(c['rep'].astype(np.float32), y)
        A_f[r] = roc_auc_score(y, pf)
        Acc_f[r] = ((pf >= 0.5).astype(int) == y).mean()

        preds = np.array(preds)
        correct = (preds == y[None, :])
        Acc_best[r] = correct.mean(1).max()
        Acc_oracle[r] = correct.any(0).mean()
        others_all_wrong = (~correct).sum(0)[None, :] == (NV - 1)
        Uni[r] = (correct & others_all_wrong).mean(1)
        for a in range(NV):
            for b in range(NV):
                Dis[r, a, b] = (preds[a] != preds[b]).mean()
                CKA[r, a, b] = 1.0 if a == b else linear_cka(feats[a], feats[b])
        print(f's{sid:2d} N={N[r]:5d} AUROC_view={A_v[r].round(3)} fus={A_f[r]:.3f} '
              f'best={Acc_best[r]:.3f} oracle={Acc_oracle[r]:.3f} uni={Uni[r].round(3)}')

    out = os.path.join(os.path.dirname(__file__), 'view_probe_results_v2.npz')
    np.savez(out, auroc_view=A_v, auroc_fusion=A_f, acc_view=Acc_v, acc_fusion=Acc_f,
             acc_best_view=Acc_best, acc_oracle=Acc_oracle, disagree=Dis, unique=Uni,
             cka=CKA, n=N, pos=P, sids=np.array(SIDS))
    print('\nsaved', out)
    print('mean view AUROC', A_v.mean(0).round(4), ' fusion', round(A_f.mean(), 4))
    print('mean unique-correct', Uni.mean(0).round(4))
    print('mean CKA off-diag', round(CKA.mean(0)[~np.eye(NV, dtype=bool)].mean(), 4))

if __name__ == '__main__':
    main()
