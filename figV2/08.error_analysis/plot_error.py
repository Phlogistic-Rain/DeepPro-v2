import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _figlib as F

DROOT = os.path.join(F.PROJ_ROOT, 'data', 'Benchmark Dataset', 'csv')
BASES = 'ACGT'
L = 81
EPS = 1e-9

METHODS = [F.HERO] + F.BASELINES
C_FP, C_FN = '#4C72B0', '#DD8452'
C_OK, C_BAD = '#4C72B0', '#C1272D'
HERO_C = F.COLORS[F.HERO]

def freq_mat(seqs):
    M = np.zeros((4, L))
    for s in seqs:
        for i, ch in enumerate(s):
            j = BASES.find(ch)
            if j >= 0:
                M[j, i] += 1
    return M / max(len(seqs), 1)

def positional_entropy(seqs):
    M = freq_mat(seqs)
    H = -(M * np.log2(M + EPS)).sum(0)
    return H.mean()

def load_all():
    sids = F.ALL_SPECIES
    per = {}
    for sid in sids:
        yt, prob = F.load_species_ensemble(F.HERO, sid, 'test')
        assert yt is not None, f'deeppro_v2 s{sid} missing draw_data'
        pred = (prob >= 0.5).astype(int)
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        n = len(yt)
        df = pd.read_csv(os.path.join(DROOT, 'Test', f'{sid}_test.csv'))
        pos = df[df.label == 1]['text'].values
        per[sid] = dict(n=n, fp=fp, fn=fn, err=(fp + fn) / n,
                        fp_rate=fp / n, fn_rate=fn / n,
                        div=positional_entropy(pos))
    return sids, per

def pooled_pred(method):
    P, Y = [], []
    for sid in F.ALL_SPECIES:
        yt, prob = F.load_species_ensemble(method, sid, 'test')
        if yt is None:
            continue
        P.append(prob); Y.append(yt)
    return np.concatenate(P), np.concatenate(Y)

def risk_coverage(prob, y, n_pts=220):
    pred = (prob >= 0.5).astype(int)
    wrong = (pred != y).astype(float)
    conf = np.maximum(prob, 1 - prob)
    order = np.argsort(-conf)
    csum = np.cumsum(wrong[order])
    n = len(y)
    ks = np.arange(1, n + 1)
    cov = ks / n
    sel_err = csum / ks
    aurc = np.trapezoid(sel_err, cov)
    idx = np.unique(np.linspace(0, n - 1, n_pts).astype(int))
    return cov[idx], sel_err[idx], aurc

def ptitle(ax, letter, text, x=-0.12, y=1.03):
    ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=x, y=y - 0.01)
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.2, fontweight='bold', va='bottom')

def main():
    F.set_style()
    sids, per = load_all()

    fig = plt.figure(figsize=(7.2, 5.55))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.90, 1.0],
                  hspace=0.72, wspace=0.40,
                  left=0.075, right=0.985, top=0.90, bottom=0.115)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    order = sorted(sids, key=lambda s: -per[s]['err'])
    x = np.arange(len(order))
    fn_r = np.array([100 * per[s]['fn_rate'] for s in order])
    fp_r = np.array([100 * per[s]['fp_rate'] for s in order])
    ax_a.bar(x, fn_r, width=0.74, color=C_FN, edgecolor='white', lw=0.3,
             label='False negative (missed promoter)', zorder=3)
    ax_a.bar(x, fp_r, bottom=fn_r, width=0.74, color=C_FP, edgecolor='white', lw=0.3,
             label='False positive', zorder=3)
    ax_a.set_xticks(x)
    labels = [f'{F.SPECIES_NAME[s]}' for s in order]
    ax_a.set_xticklabels(labels, rotation=52, ha='right', fontsize=5.4, fontstyle='italic')
    worst3 = set(order[:3])
    for tl, s in zip(ax_a.get_xticklabels(), order):
        if s in worst3:
            tl.set_color('#B5521E'); tl.set_fontweight('bold')
    ax_a.set_xlim(-0.7, len(order) - 0.3)
    ax_a.set_ylabel('Per-species error rate (%)', fontsize=7.2)
    ax_a.set_ylim(0, max(fn_r + fp_r) * 1.16)
    ax_a.spines[['top', 'right']].set_visible(False)
    ax_a.yaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=6.0, loc='upper right', handlelength=1.0,
                handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)
    ptitle(ax_a, 'a', 'Errors concentrate in a few high-diversity species; both error types present',
           x=-0.052)

    prob, y = pooled_pred(F.HERO)
    pred = (prob >= 0.5).astype(int)
    correct = pred == y
    bins = np.linspace(0, 1, 41)
    for data, c, lab in [
        (prob[correct],  C_OK,  f'Correct (n={int(correct.sum()):,})'),
        (prob[~correct], C_BAD, f'Wrong (n={int((~correct).sum()):,})'),
    ]:
        ax_b.hist(data, bins=bins, density=True, histtype='stepfilled',
                  color=c, alpha=0.13, lw=0)
        ax_b.hist(data, bins=bins, density=True, histtype='step',
                  color=c, lw=1.3, label=lab)
    ax_b.axvline(0.5, color='#888888', lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax_b.text(0.5, 0.02, 'decision\nboundary', transform=ax_b.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=5.3, color='#7a7a7a', linespacing=0.95)
    ax_b.set_xlabel('Predicted promoter probability', fontsize=7.2)
    ax_b.set_ylabel('Density', fontsize=7.2)
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.legend(fontsize=5.8, loc='upper center', handlelength=1.0, handletextpad=0.4,
                borderaxespad=0.3)
    ax_b.spines[['top', 'right']].set_visible(False)
    ptitle(ax_b, 'b', 'Errors hug the boundary')

    aurc = {}
    COV_FLOOR = 0.05
    for m in METHODS:
        pm, ym = pooled_pred(m)
        cov, se, a = risk_coverage(pm, ym)
        aurc[m] = a
        vis = cov >= COV_FLOOR
        is_hero = (m == F.HERO)
        ax_c.plot(100 * cov[vis], 100 * se[vis], color=F.COLORS[m],
                  lw=2.0 if is_hero else 1.1,
                  zorder=5 if is_hero else 3,
                  label=f'{F.DISPLAY[m]} ({100*a:.1f})')
    ax_c.set_xlabel('Coverage (% most-confident retained)', fontsize=7.2)
    ax_c.set_ylabel('Selective error rate (%)', fontsize=7.2)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0, None)
    ax_c.spines[['top', 'right']].set_visible(False)
    ax_c.yaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax_c.set_axisbelow(True)
    leg = ax_c.legend(title='Method (AURC)', fontsize=5.5, title_fontsize=5.8,
                      loc='upper left', handlelength=1.2, handletextpad=0.5,
                      borderaxespad=0.3, labelspacing=0.28)
    leg._legend_box.align = 'left'
    ax_c.text(0.97, 0.05, 'lower = safer', transform=ax_c.transAxes, ha='right', va='bottom',
              fontsize=5.6, fontstyle='italic', color='#666666')
    ptitle(ax_c, 'c', 'Fewest confident errors (selective prediction)')

    div = np.array([per[s]['div'] for s in sids])
    err = np.array([100 * per[s]['err'] for s in sids])
    ax_d.scatter(div, err, s=34, color='#4C72B0', edgecolor='#2b2b2b',
                 linewidth=0.5, zorder=3)
    b1, b0 = np.polyfit(div, err, 1)
    xx = np.linspace(div.min(), div.max(), 50)
    ax_d.plot(xx, b1 * xx + b0, color='#444444', lw=1.1, ls=(0, (4, 2)), zorder=2)
    rho, pr = spearmanr(div, err)
    rp, _ = pearsonr(div, err)
    ax_d.text(0.04, 0.94,
              f'Spearman $\\rho$ = {rho:.2f} ($P$ = {pr:.3f})\nPearson $r$ = {rp:.2f},  $n$ = 23',
              transform=ax_d.transAxes, fontsize=5.9, va='top', ha='left')
    for s, dy in [(15, 4), (10, -9)]:
        i = sids.index(s)
        ax_d.annotate(F.SPECIES_NAME[s], (div[i], err[i]),
                      textcoords='offset points', xytext=(-5, dy), ha='right',
                      fontsize=5.6, fontstyle='italic', color='#1a1a1a')
    ax_d.set_xlabel('Promoter positional entropy (bits)', fontsize=7.2)
    ax_d.set_ylabel('Per-species error rate (%)', fontsize=7.2)
    ax_d.spines[['top', 'right']].set_visible(False)
    ax_d.yaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax_d.set_axisbelow(True)
    ptitle(ax_d, 'd', 'Harder where promoters are diverse')

    F.save_panels(fig, [
        ('a_fp_fn_breakdown',   ax_a, []),
        ('b_confidence_hist',   ax_b, []),
        ('c_risk_coverage',     ax_c, []),
        ('d_error_vs_diversity', ax_d, []),
    ], os.path.join(HERE, 'panel'))

    out = os.path.join(HERE, 'fig_error_analysis')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)

    tot_fp = sum(per[s]['fp'] for s in sids)
    tot_fn = sum(per[s]['fn'] for s in sids)
    tot_n = sum(per[s]['n'] for s in sids)
    print(f'saved {out}.png / .pdf  (+ 4 panels)')
    print(f'[FP/FN balance] FP={tot_fp} FN={tot_fn} ratio(FP/FN)={tot_fp/tot_fn:.2f} '
          f'overall err={100*(tot_fp+tot_fn)/tot_n:.2f}%')
    print(f'[risk-coverage AURC x100, lower=safer] ' +
          '  '.join(f'{F.DISPLAY[m]}={100*aurc[m]:.2f}' for m in METHODS))
    lowest = min(aurc, key=aurc.get)
    print(f'   -> lowest AURC = {F.DISPLAY[lowest]}  (DeepPro-v2 safest overall: '
          f'{aurc[F.HERO] == aurc[lowest]})')
    worst = max(sids, key=lambda s: per[s]['err'])
    print(f'[worst species] s{worst} {F.SPECIES_NAME[worst]} err={100*per[worst]["err"]:.2f}%')
    print(f'[error~diversity] Spearman rho={rho:.3f} P={pr:.4f} | Pearson r={rp:.3f}')
    print(f'[confidence] correct mean p-dist-from-0.5={np.abs(prob[correct]-0.5).mean():.3f} '
          f'wrong={np.abs(prob[~correct]-0.5).mean():.3f}')

if __name__ == '__main__':
    main()
