import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _figlib as F
F.set_style()

METHODS = ['DeepPro-v2', 'msBERT', 'DNABERT2-CAMP', 'GROVER', 'HyenaDNA',
           'iPro-MP', 'Prompt', 'PromoterLCNN', 'iPro-WAEL']
HERO = 'DeepPro-v2'
BEST_BASELINE = 'msBERT'
HERO_RED = '#C1272D'
WIN_RED = '#C1272D'
LOSS_BLUE = '#0072B2'

COLORS = {
    'DeepPro-v2':    '#C1272D',
    'msBERT':        '#0072B2',
    'iPro-MP':       '#E69F00',
    'DNABERT2-CAMP': '#56B4E9',
    'HyenaDNA':      '#009E73',
    'Prompt':        '#CC79A7',
    'iPro-WAEL':     '#948B3D',
    'GROVER':        '#7F7F7F',
    'PromoterLCNN':  '#8C564B',
}
DISPLAY = {m: m for m in METHODS}

Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

def holm_correct(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj

def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

def load_matrix():
    wb = openpyxl.load_workbook(os.path.join(HERE, '..', '..', 'paper', 'Result.xlsx'),
                                data_only=True)
    ws = wb['test']
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    import re
    colmap = {}
    for c, v in enumerate(hdr):
        if v and isinstance(v, str) and '(' in v:
            colmap[re.search(r'\(([^)]+)\)', v).group(1).strip()] = c
    M, sids = [], []
    for r in rows[2:]:
        if r[1] is None or r[1] == 'AVG' or not isinstance(r[0], (int, float)):
            continue
        sids.append(int(r[0]))
        M.append([r[colmap[m] + 5] for m in METHODS])
    return np.array(M, float), sids

def compute_stats(M):
    N, k = M.shape

    ranks = np.array([rankdata(-M[i]) for i in range(N)])
    avg_rank = ranks.mean(0)
    fr = friedmanchisquare(*[M[:, j] for j in range(k)])
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))

    pair_p = np.ones((k, k))
    raw, idxpairs = [], []
    for a in range(k):
        for b in range(a + 1, k):
            try:
                _, p = wilcoxon(M[:, a], M[:, b])
            except ValueError:
                p = 1.0
            raw.append(p); idxpairs.append((a, b))
    adj = holm_correct(raw)
    for (a, b), p in zip(idxpairs, adj):
        pair_p[a, b] = pair_p[b, a] = p

    dp, ms = METHODS.index(HERO), METHODS.index(BEST_BASELINE)
    d = M[:, dp] - M[:, ms]
    w = wilcoxon(M[:, dp], M[:, ms])
    wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())

    order = np.argsort(avg_rank)

    summary = {
        'N_species': N, 'k_methods': k,
        'friedman_chi2': float(fr.statistic), 'friedman_p': float(fr.pvalue),
        'CD_0.05': float(cd), 'q_alpha_0.05': Q05[k],
        'avg_rank': {METHODS[i]: float(avg_rank[i]) for i in range(k)},
        'per_species_mean_MCC': {METHODS[i]: float(M[:, i].mean()) for i in range(k)},
        'pairwise_wilcoxon_holm_p': {
            f'{METHODS[a]}__vs__{METHODS[b]}': float(pair_p[a, b])
            for a, b in idxpairs},
        'DeepPro-v2_vs_holm_p': {
            METHODS[b]: float(pair_p[dp, b]) for b in range(k) if b != dp},
        'v2_vs_msBERT': {
            'wilcoxon_stat': float(w.statistic), 'wilcoxon_p': float(w.pvalue),
            'wins': wins, 'losses': losses, 'ties': ties,
            'dMCC_mean': float(d.mean()), 'dMCC_median': float(np.median(d)),
            'dMCC_min': float(d.min()), 'dMCC_max': float(d.max())},
    }
    return {'avg_rank': avg_rank, 'cd': cd, 'fr': fr, 'pair_p': pair_p,
            'order': order, 'd': d, 'w': w, 'wins': wins, 'losses': losses,
            'summary': summary}

def draw_cd(ax, avg_rank, cd, fr_p):
    k = len(METHODS)
    order = np.argsort(avg_rank)
    lo, hi = 1, k
    ax.set_xlim(lo - 3.3, hi + 3.3)
    ax.set_ylim(0.0, 1.0)
    axis_y = 0.47

    ax.plot([lo, hi], [axis_y, axis_y], color='#333333', lw=1.0, zorder=2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y, axis_y - 0.028], color='#333333', lw=0.8, zorder=2)
        ax.text(r, axis_y - 0.05, str(r), ha='center', va='top', fontsize=6.2)

    sr = order
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (avg_rank[sr[j + 1]] - avg_rank[sr[i]]) < cd:
            j += 1
        if j > i:
            cliques.append((sr[i], sr[j]))
        i += 1
    cliques = [c for c in cliques if not any(
        c != e and avg_rank[e[0]] <= avg_rank[c[0]] and avg_rank[c[1]] <= avg_rank[e[1]]
        for e in cliques)]
    bar_y = axis_y + 0.05
    for a, b in cliques:
        ra, rb = avg_rank[a], avg_rank[b]
        ax.plot([min(ra, rb) - 0.06, max(ra, rb) + 0.06], [bar_y, bar_y],
                color='#8A8A8A', lw=2.6, solid_capstyle='round', zorder=4)
        bar_y += 0.048
    ax.text(hi + 0.15, axis_y + 0.06, 'connected = not sig.\n(Nemenyi, $\\alpha$=0.05)',
            fontsize=5.4, color='#8A8A8A', ha='left', va='bottom', style='italic')

    cdy = 0.80
    ax.plot([lo, lo + cd], [cdy, cdy], color=HERO_RED, lw=1.8, zorder=4)
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [cdy - 0.026, cdy + 0.026], color=HERO_RED, lw=1.0, zorder=4)
    ax.text(lo + cd / 2, cdy + 0.042, f'CD = {cd:.2f}', ha='center', va='bottom',
            fontsize=6.8, color=HERO_RED, fontweight='bold')

    half = (k + 1) // 2
    left_idx = list(order[:half])
    right_idx = list(order[half:][::-1])
    top_ly = axis_y - 0.10
    step = 0.082
    ly_left = [top_ly - step * t for t in range(len(left_idx))]
    ly_right = [top_ly - step * t for t in range(len(right_idx))]

    def elbow(mi, ly, side):
        r = avg_rank[mi]
        m = METHODS[mi]
        is_dp = (m == HERO)
        col = COLORS[m]
        x_text = lo - 0.55 if side == 'left' else hi + 0.55
        ha = 'right' if side == 'left' else 'left'
        lw = 1.7 if is_dp else 0.9
        ax.plot([r, r], [axis_y, ly], color=col, lw=lw, zorder=3)
        ax.plot([r, x_text], [ly, ly], color=col, lw=lw, zorder=3)
        ax.text(x_text + (-0.08 if side == 'left' else 0.08), ly,
                f'{DISPLAY[m]}  ({r:.2f})', ha=ha, va='center',
                fontsize=6.6, fontweight='bold' if is_dp else 'normal',
                color=col if is_dp else '#1a1a1a')
        ax.scatter([r], [axis_y], s=34 if is_dp else 16, color=col, zorder=6,
                   edgecolor='white', lw=0.6)

    for mi, ly in zip(left_idx, ly_left):
        elbow(mi, ly, 'left')
    for mi, ly in zip(right_idx, ly_right):
        elbow(mi, ly, 'right')

    ax.text((lo + hi) / 2, min(ly_left + ly_right) - 0.07, 'Average rank  (1 = best)',
            ha='center', va='top', fontsize=7.2, fontweight='bold')
    ax.axis('off')
    ax.set_title('a', loc='left', fontsize=11, fontweight='bold', x=-0.055, y=0.99)
    ax.text(-0.02, 1.0, f'Friedman–Nemenyi critical-difference diagram   '
            f'(Friedman $P$ = {fr_p:.1e})',
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

def draw_pmatrix(ax, P, order):
    k = len(order)
    Po = P[np.ix_(order, order)]
    L = -np.log10(np.clip(Po, 1e-6, 1))
    mask = np.triu(np.ones_like(L, bool))
    Lm = np.ma.array(L, mask=mask)
    im = ax.imshow(Lm, cmap='Reds', vmin=0, vmax=4.0, aspect='equal')
    names = [DISPLAY[METHODS[i]] for i in order]
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=5.4)
    ax.set_yticklabels(names, fontsize=5.4)
    for a in range(k):
        for b in range(a):
            s = stars(Po[a, b])
            ax.text(b, a, s, ha='center', va='center', fontsize=5.4,
                    color='white' if L[a, b] > 2.0 else '#333333',
                    fontweight='bold' if s != 'ns' else 'normal')
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() == HERO:
            lab.set_color(HERO_RED); lab.set_fontweight('bold')
    cax = ax.inset_axes([1.04, 0.0, 0.04, 1.0])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$-\log_{10}\,P_{\mathrm{Holm}}$', fontsize=6.2)
    cb.ax.tick_params(labelsize=5.6)
    ax.set_title('b', loc='left', fontsize=11, fontweight='bold', x=-0.28, y=1.02)
    ax.text(0.0, 1.04, 'Pairwise Wilcoxon signed-rank',
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0.0, -0.40, '*** P<0.001   ** P<0.01   * P<0.05   ns = not sig.',
            transform=ax.transAxes, fontsize=5.6, color='#666666')
    return cax

def draw_delta(ax, d, sids, wins, losses):
    o = np.argsort(d)
    y = np.arange(len(d))
    cols = [WIN_RED if d[i] > 0 else LOSS_BLUE for i in o]
    ax.barh(y, d[o], color=cols, edgecolor='white', lw=0.3, height=0.72, zorder=3)
    ax.axvline(0, color='#333333', lw=0.7)
    labels = [f'{F.SPECIES_NAME[sids[i]]} ({sids[i]})' for i in o]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.2, fontstyle='italic')
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(r'$\Delta$MCC  (DeepPro-v2 $-$ msBERT)', fontsize=7.0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.text(0.97, 0.04, f'DeepPro-v2 wins {wins}/{wins + losses}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6.4, fontweight='bold', color=HERO_RED)
    ax.set_title('c', loc='left', fontsize=11, fontweight='bold', x=-0.34, y=1.02)
    ax.text(0.0, 1.02, 'Per-species gain over best baseline',
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

def main():
    M, sids = load_matrix()
    R = compute_stats(M)
    S = R['summary']

    with open(os.path.join(HERE, 'stats_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(S, f, indent=2, ensure_ascii=False)

    print('Friedman chi2=%.2f  P=%.2e   CD=%.3f' % (
        S['friedman_chi2'], S['friedman_p'], S['CD_0.05']))
    for m in sorted(METHODS, key=lambda x: S['avg_rank'][x]):
        print('  %-14s avg_rank=%.3f' % (m, S['avg_rank'][m]))
    vv = S['v2_vs_msBERT']
    print('Wilcoxon W=%.0f P=%.2e  wins=%d/%d  (dMCC mean=%.4f median=%.4f)' % (
        vv['wilcoxon_stat'], vv['wilcoxon_p'], vv['wins'], vv['wins'] + vv['losses'],
        vv['dMCC_mean'], vv['dMCC_median']))

    fig = plt.figure(figsize=(7.4, 6.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.48],
                  width_ratios=[1.0, 1.05], hspace=0.30, wspace=0.60,
                  left=0.085, right=0.95, top=0.93, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_cd(ax_a, R['avg_rank'], R['cd'], S['friedman_p'])
    cax_b = draw_pmatrix(ax_b, R['pair_p'], R['order'])
    draw_delta(ax_c, R['d'], sids, R['wins'], R['losses'])

    F.save_panels(fig, [
        ('a_critical_difference', ax_a, []),
        ('b_pairwise_wilcoxon',   ax_b, [cax_b]),
        ('c_per_species_delta',   ax_c, []),
    ], os.path.join(HERE, 'panel'))

    out = os.path.join(HERE, 'fig_significance')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', out + '.png / .pdf  + panel/  + stats_summary.json')

if __name__ == '__main__':
    main()
