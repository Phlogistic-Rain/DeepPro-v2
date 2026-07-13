import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr, t as tdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _figlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
DROOT = os.path.join(F.PROJ_ROOT, 'data', 'Benchmark Dataset', 'csv', 'Test')
BASES = 'ACGT'
L = 81
SIDS = list(range(1, 24))
ARCHAEA = {11, 21}
HERO = 10

POS_C = '#C1272D'
ARCH_C = '#1f9a9c'
BACT_C = '#8593A5'
EDGE_C = '#2b2b2b'
FIT_C = POS_C
GRID_C = '#EEEEEE'
RNG = np.random.default_rng(0)

def freq_mat(seqs):
    M = np.zeros((4, L))
    for s in seqs:
        for i, ch in enumerate(s):
            j = BASES.find(ch)
            if j >= 0:
                M[j, i] += 1
    return M / max(len(seqs), 1)

def mean_pairwise_dist(seqs):
    M = freq_mat(seqs)
    return (1.0 - (M ** 2).sum(0)).mean()

def probe_auc(rep, y):
    X = StandardScaler().fit_transform(rep.astype(np.float32))
    return cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                           X, y.astype(int), cv=5, scoring='roc_auc').mean()

def compute_table():
    D, S, MCC, N = [], [], [], []
    for sid in SIDS:
        d = F.load_repr_cache(sid)
        df = pd.read_csv(os.path.join(DROOT, f'{sid}_test.csv'))
        pos = df[df.label == 1]['text'].values
        D.append(mean_pairwise_dist(pos))
        S.append(probe_auc(d['rep'], d['y']))
        yt, prob = F.load_species_ensemble('deeppro_v2', sid, 'test')
        yp = (prob >= 0.5).astype(int)
        MCC.append(_mcc(yt, yp))
        N.append(len(d['y']))
    return (np.array(D), np.array(S), np.array(MCC), np.array(N))

def _mcc(y, yp):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y, yp)

def pstr(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return rf'{mant:.1f}\times10^{{{exp}}}'
    return f'{p:.3f}'

def boot_ci_spearman(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        rs.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return lo, hi

def partial_spearman(x, y, z):
    rxy = spearmanr(x, y)[0]
    rxz = spearmanr(x, z)[0]
    ryz = spearmanr(y, z)[0]
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    dfree = n - 3
    tval = r * np.sqrt(dfree / (1 - r ** 2))
    p = 2 * tdist.sf(abs(tval), dfree)
    return r, p, rxy

def boot_ci_partial(x, y, z, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r, _, _ = partial_spearman(x[idx], y[idx], z[idx])
            if np.isfinite(r):
                rs.append(r)
        except Exception:
            continue
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return lo, hi

def pt_color(sid):
    if sid == HERO:
        return POS_C
    if sid in ARCHAEA:
        return ARCH_C
    return BACT_C

def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color=GRID_C, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

def ptitle(ax, letter, text, x=-0.13, y=1.02):
    ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=x, y=y - 0.015)
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.2,
            fontweight='bold', va='bottom')

def scatter_fit(ax, xv, yv, colors, xlabel, ylabel, annot_species, ann_text,
                ann_xy=(0.035, 0.045), ann_va='bottom'):
    ax.scatter(xv, yv, c=colors, s=48, edgecolor=EDGE_C, linewidth=0.5, zorder=3)
    b, a = np.polyfit(xv, yv, 1)
    xx = np.linspace(xv.min(), xv.max(), 60)
    ax.plot(xx, b * xx + a, color=FIT_C, lw=1.4, ls=(0, (4, 2)), zorder=2)
    ax.text(ann_xy[0], ann_xy[1], ann_text, transform=ax.transAxes,
            fontsize=6.3, va=ann_va, ha='left', linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#cccccc', lw=0.5, alpha=0.9))
    xlo, xhi = xv.min(), xv.max()
    span = xhi - xlo
    for sid in annot_species:
        i = SIDS.index(sid)
        right = xv[i] > xlo + 0.66 * span
        dx, ha = (-6, 'right') if right else (6, 'left')
        ax.annotate(F.SPECIES_NAME[sid], (xv[i], yv[i]), textcoords='offset points',
                    xytext=(dx, 4), ha=ha, fontsize=5.7, fontstyle='italic', color='#1a1a1a')
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    style_ax(ax)

def main():
    F.set_style()
    D, S, MCC, N = compute_table()
    logN = np.log10(N)
    colors = [pt_color(s) for s in SIDS]

    r_ds, p_ds = spearmanr(D, S)
    ci_ds = boot_ci_spearman(D, S)
    r_dm, p_dm = spearmanr(D, MCC)
    ci_dm = boot_ci_spearman(D, MCC, seed=2)
    r_dn, p_dn = spearmanr(D, N)
    pr, pp, zero = partial_spearman(D, S, logN)
    ci_pr = boot_ci_partial(D, S, logN)

    fig = plt.figure(figsize=(7.2, 6.7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.02],
                  hspace=0.40, wspace=0.30,
                  left=0.095, right=0.975, top=0.935, bottom=0.115)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    xlab_div = 'Promoter diversity\n(mean pairwise distance, per bp)'

    ann_a = (rf'Spearman $\rho$ = {r_ds:.2f}' + '\n'
             + rf'$P$ = ${pstr(p_ds)}$   ($n$ = 23)' + '\n'
             + rf'95% CI [{ci_ds[0]:.2f}, {ci_ds[1]:.2f}]')
    scatter_fit(ax_a, D, S, colors, xlab_div,
                'Representation linear-probe\nAUROC (5-fold CV)',
                [HERO, 15, 14, 17], ann_a, ann_xy=(0.035, 0.045), ann_va='bottom')

    ann_b = (rf'Spearman $\rho$ = {r_dm:.2f}' + '\n'
             + rf'$P$ = ${pstr(p_dm)}$   ($n$ = 23)' + '\n'
             + rf'95% CI [{ci_dm[0]:.2f}, {ci_dm[1]:.2f}]')
    scatter_fit(ax_b, D, MCC, colors, xlab_div,
                'Ensemble test MCC',
                [HERO, 15, 4], ann_b, ann_xy=(0.035, 0.045), ann_va='bottom')

    order = np.argsort(D)
    xpos = np.arange(len(SIDS))
    bar_colors = [pt_color(SIDS[i]) for i in order]
    ax_c.bar(xpos, D[order], color=bar_colors, edgecolor='white', lw=0.3, width=0.76, zorder=3)
    ax_c.axhline(0.75, color='#555555', lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax_c.text(len(SIDS) - 0.4, 0.7515, 'max diversity (0.75)', fontsize=5.3,
              ha='right', va='bottom', color='#555555')
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([f'{F.SPECIES_NAME[SIDS[i]]}' for i in order],
                         rotation=60, ha='right', fontsize=5.2, fontstyle='italic')
    ax_c.set_ylim(0.60, 0.775)
    ax_c.set_ylabel('Promoter diversity\n(mean pairwise distance, per bp)', fontsize=7)
    ax_c.set_xlim(-0.7, len(SIDS) - 0.3)
    style_ax(ax_c)
    ax_c.text(0.02, 0.965,
              r'$D=\dfrac{1}{L}\sum_{i=1}^{L}\left(1-\sum_{b}p_{b,i}^{2}\right)$'
              + '\nmean per-bp pairwise mismatch\namong promoter (positive) sequences',
              transform=ax_c.transAxes, fontsize=5.9, va='top', ha='left', linespacing=1.4,
              bbox=dict(boxstyle='round,pad=0.32', fc='white', ec='#cccccc', lw=0.5, alpha=0.92))
    ax_c.legend(handles=[Patch(fc=POS_C, label='E. coli'),
                         Patch(fc=ARCH_C, label='Archaea'),
                         Patch(fc=BACT_C, label='Other bacteria')],
                loc='lower right', fontsize=5.8, handlelength=1.0, borderpad=0.3,
                labelspacing=0.28)

    resD = D - np.polyval(np.polyfit(logN, D, 1), logN)
    resS = S - np.polyval(np.polyfit(logN, S, 1), logN)
    ax_d.scatter(resD, resS, c=colors, s=48, edgecolor=EDGE_C, linewidth=0.5, zorder=3)
    bb, aa = np.polyfit(resD, resS, 1)
    xx = np.linspace(resD.min(), resD.max(), 60)
    ax_d.plot(xx, bb * xx + aa, color=FIT_C, lw=1.4, ls=(0, (4, 2)), zorder=2)
    ann_d = (rf'Partial Spearman $\rho$ = {pr:.2f}' + '\n'
             + rf'$P$ = ${pstr(pp)}$  (control $n$)' + '\n'
             + rf'95% CI [{ci_pr[0]:.2f}, {ci_pr[1]:.2f}]' + '\n'
             + rf'zero-order $\rho$ = {zero:.2f};  $\rho$(div,$n$) = {r_dn:.2f}')
    ax_d.text(0.035, 0.045, ann_d, transform=ax_d.transAxes, fontsize=6.0,
              va='bottom', ha='left', linespacing=1.35,
              bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#cccccc', lw=0.5, alpha=0.9))
    for sid in [HERO, 15]:
        i = SIDS.index(sid)
        ax_d.annotate(F.SPECIES_NAME[sid], (resD[i], resS[i]), textcoords='offset points',
                      xytext=(6, 4), ha='left', fontsize=5.7, fontstyle='italic', color='#1a1a1a')
    ax_d.axhline(0, color='#cfcfcf', lw=0.6, zorder=1)
    ax_d.axvline(0, color='#cfcfcf', lw=0.6, zorder=1)
    ax_d.set_xlabel('Promoter diversity | $n$  (residual)', fontsize=7)
    ax_d.set_ylabel('Separability (AUROC) | $n$\n(residual)', fontsize=7)
    style_ax(ax_d)

    ptitle(ax_a, 'a', 'Diversity vs separability', x=-0.155)
    ptitle(ax_b, 'b', 'Diversity vs ensemble MCC', x=-0.135)
    ptitle(ax_c, 'c', 'Per-species promoter diversity', x=-0.115)
    ptitle(ax_d, 'd', 'Holds after controlling sample size', x=-0.155)

    save_dir = os.path.join(HERE, 'panel')
    F.save_panels(fig, [
        ('a_diversity_vs_separability', ax_a, []),
        ('b_diversity_vs_mcc', ax_b, []),
        ('c_diversity_definition', ax_c, []),
        ('d_partial_correlation', ax_d, []),
    ], save_dir)

    out = os.path.join(HERE, 'fig_promoter_diversity')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)

    eco = SIDS.index(HERO)
    print(f'saved {out}.png / .pdf  (+ 4 panels in {save_dir})')
    print(f'Diversity D range: [{D.min():.3f}, {D.max():.3f}]  (theoretical max 0.75)')
    print(f'E. coli D={D[eco]:.3f}  rank {int(np.sum(D <= D[eco]))}/23 (1=least diverse), '
          f'AUROC={S[eco]:.3f}, MCC={MCC[eco]:.3f}')
    print(f'Spearman  diversity~separability: rho={r_ds:.3f}  P={p_ds:.4g}  '
          f'95%CI[{ci_ds[0]:.3f},{ci_ds[1]:.3f}]')
    print(f'Spearman  diversity~MCC        : rho={r_dm:.3f}  P={p_dm:.4g}  '
          f'95%CI[{ci_dm[0]:.3f},{ci_dm[1]:.3f}]')
    print(f'Spearman  diversity~N (confound): rho={r_dn:.3f}  P={p_dn:.4g}')
    print(f'Partial   diversity~separability | log10(N): rho={pr:.3f}  P={pp:.4g}  '
          f'95%CI[{ci_pr[0]:.3f},{ci_pr[1]:.3f}]  (zero-order {zero:.3f})')

if __name__ == '__main__':
    main()
