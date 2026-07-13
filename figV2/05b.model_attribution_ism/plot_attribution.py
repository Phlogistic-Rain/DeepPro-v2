import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'figV1'))
sys.path.insert(0, os.path.join(_ROOT, 'figV1', '09.sequence_motif'))
from _figlib import set_style, save_panels
import plot_motif as pm

BASES = 'ACGT'
TSS = pm.TSS
X0, X1 = pm.X0, pm.X1
BOX35, BOX10 = pm.BOX35, pm.BOX10
NT_COLORS = {'A': '#2E8B57', 'C': '#1F6FB2', 'G': '#E69F00', 'T': '#C0392B'}

def main():
    set_style()
    d = np.load(os.path.join(_HERE, 'ism_v2.npz'), allow_pickle=True)
    importance = d['importance']; heat = d['heat']; n_seq = int(d['n_seq'])

    eco = pm.load_species(10)
    fp, fn = pm.pos_neg_freq(eco)
    kl = pm.kl_profile(fp, fn)

    rng = np.arange(X0, X1)
    r_all, p_all = pearsonr(importance, kl)
    rho_all, _ = spearmanr(importance, kl)
    r_win, _ = pearsonr(importance[X0:X1], kl[X0:X1])

    def mm(v):
        v = v[X0:X1]
        return (v - v.min()) / (np.ptp(v) + 1e-12)
    imp_n, kl_n = mm(importance), mm(kl)

    fig = plt.figure(figsize=(7.1, 6.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 1.15],
                  hspace=0.30, left=0.135, right=0.905, top=0.92, bottom=0.10)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

    imp_w = importance[X0:X1]
    ax_a.fill_between(rng, imp_w, color='#7B4FA0', alpha=0.30, lw=0, zorder=2)
    ax_a.plot(rng, imp_w, color='#5B3A80', lw=1.1, zorder=3)
    ax_a.axhline(0, color='#888888', lw=0.5)
    ax_a.set_ylabel('Model importance\n(Δ promoter prob.)', fontsize=7)
    ax_a.set_xlim(X0 - 0.5, X1 - 0.5)
    ax_a.set_ylim(min(0, imp_w.min() * 1.15), imp_w.max() * 1.28)
    ax_a.set_xticks([]); ax_a.spines[['top', 'right']].set_visible(False)

    ax_b.plot(rng, imp_n, color='#5B3A80', lw=1.3, label='Model attribution (ISM)', zorder=3)
    ax_b.plot(rng, kl_n, color='#34495E', lw=1.1, ls=(0, (3, 1.5)), label='Data information (KL)', zorder=3)
    ax_b.set_ylabel('Normalized\nsignal (0–1)', fontsize=7)
    ax_b.set_xlim(X0 - 0.5, X1 - 0.5); ax_b.set_ylim(-0.03, 1.45)
    ax_b.set_xticks([]); ax_b.spines[['top', 'right']].set_visible(False)
    ax_b.legend(fontsize=6.2, loc='upper left', handlelength=1.6, ncol=2, columnspacing=1.0, borderaxespad=0.2)
    ax_b.text(0.985, 0.985,
              f'Pearson $r$ = {r_all:.2f}  (window {r_win:.2f})\nSpearman $\\rho$ = {rho_all:.2f}  ($n$ = 81)',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=6.3,
              bbox=dict(boxstyle='round,pad=0.3', fc='#f4f1f8', ec='#b9a9d0', lw=0.5))

    hw = heat[:, X0:X1]
    vmax = np.percentile(np.abs(hw), 99.0)
    im = ax_c.imshow(hw, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                     extent=[X0 - 0.5, X1 - 0.5, 3.5, -0.5], interpolation='nearest')
    ax_c.set_yticks(range(4)); ax_c.set_yticklabels(list(BASES), fontsize=7)
    for t, b in zip(ax_c.get_yticklabels(), BASES):
        t.set_color(NT_COLORS[b]); t.set_fontweight('bold')
    ax_c.set_ylabel('Mutant base', fontsize=7)
    show_rel = [-45, -35, -25, -15, -10, -5, 1, 5]
    show_idx = [TSS + (r if r < 0 else r - 1) for r in show_rel]
    show_idx = [i for i in show_idx if X0 <= i < X1]
    ax_c.set_xticks(show_idx)
    ax_c.set_xticklabels([f'{pm.rel(i):+d}' for i in show_idx], fontsize=6.3)
    ax_c.set_xlabel('Position relative to transcription start site (+1)', fontsize=7.5)
    cax = fig.add_axes([0.912, 0.10, 0.016, 0.22])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Δ promoter prob.\n(mutant − WT)', fontsize=6.0); cb.ax.tick_params(labelsize=5.6)

    for ax in (ax_a, ax_b, ax_c):
        for (lo, hi), c in [(BOX35, '#888888'), (BOX10, '#C1272D')]:
            ax.axvspan(lo - 0.5, hi - 0.5, color=c, alpha=0.10, lw=0, zorder=0)
        ax.axvline(TSS, color='#2E8B57', lw=0.8, ls=(0, (3, 2)), zorder=1, alpha=0.8)

    yt = ax_a.get_ylim()[1] * 0.92
    ax_a.text((BOX10[0] + BOX10[1]) / 2 - 0.5, yt, '–10 box', ha='center', va='top',
              fontsize=6.5, color='#C1272D', fontweight='bold')
    ax_a.text((BOX35[0] + BOX35[1]) / 2 - 0.5, yt, '–35 box', ha='center', va='top',
              fontsize=6.5, color='#555555', fontweight='bold')
    ax_a.text(TSS, yt, '+1', ha='center', va='top', fontsize=6.5, color='#2E8B57', fontweight='bold')

    def ptitle(ax, letter, text, y=1.02):
        ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=-0.115, y=y - 0.02)
        ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
    ptitle(ax_a, 'a', f'DeepPro-v2 attribution by in-silico mutagenesis — E. coli ($n$ = {n_seq})')
    ptitle(ax_b, 'b', 'Deployed model attends the data’s discriminative −10 / +1 positions')
    ptitle(ax_c, 'c', 'Per-base mutation effect (Δ promoter probability)')
    fig.text(0.905, 0.955, 'read out from FM experts + head (not the deployed grammar PWM)',
             ha='right', va='bottom', fontsize=5.8, color='#777777', fontstyle='italic')

    save_panels(fig, [
        ('a_positional_saliency', ax_a, []),
        ('b_saliency_vs_data_kl', ax_b, []),
        ('c_example_ism_map', ax_c, [cax]),
    ], os.path.join(_HERE, 'panel'))

    out = os.path.join(_HERE, 'fig_model_attribution_ism')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', out + '.png')
    print(f'Pearson r(81)={r_all:.3f} p={p_all:.1e} | window={r_win:.3f} | Spearman rho={rho_all:.3f}')

if __name__ == '__main__':
    main()
