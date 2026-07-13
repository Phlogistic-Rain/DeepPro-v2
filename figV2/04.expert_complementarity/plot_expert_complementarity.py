import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _figlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'view_probe_results_v2.npz')
VS = ['DB-v2' if s == 'DB-2' else s for s in F.VIEW_SHORT]
VC = F.VIEW_COLORS
FUS = F.FUSION_COLOR
NV = 5

def ptitle(ax, letter, text, x=-0.15, y=1.02):
    ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=x, y=y - 0.02)
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.3, fontweight='bold', va='bottom')

def main():
    F.set_style()
    d = np.load(RES)
    Av, Af = d['auroc_view'], d['auroc_fusion']
    sids = d['sids']
    Dis = d['disagree'].mean(0)
    Uni = d['unique'].mean(0)
    CKA = d['cka'].mean(0)
    accb = d['acc_best_view']; accf = d['acc_fusion']; acco = d['acc_oracle']

    fig = plt.figure(figsize=(7.4, 8.3))

    ax_a = fig.add_axes([0.135, 0.075, 0.32, 0.85])
    order = np.argsort(Af)
    M = np.column_stack([Av[order], Af[order]])
    rows = [f'{F.SPECIES_NAME[int(sids[i])]} ({int(sids[i])})' for i in order]
    im = ax_a.imshow(M, aspect='auto', cmap='viridis', vmin=0.5, vmax=1.0)
    ax_a.set_xticks(range(6))
    ax_a.set_xticklabels(VS + ['Fusion'], rotation=40, ha='right', fontsize=6.0)
    ax_a.get_xticklabels()[5].set_color(FUS); ax_a.get_xticklabels()[5].set_fontweight('bold')
    ax_a.set_yticks(range(23)); ax_a.set_yticklabels(rows, fontsize=5.6, fontstyle='italic')
    ax_a.tick_params(length=0)
    ax_a.axvline(4.5, color='white', lw=1.8)
    for r in range(23):
        bv = int(np.argmax(M[r, :5]))
        ax_a.plot(bv, r, marker='*', ms=6.0, mfc='white', mec='black', mew=0.6, zorder=5, clip_on=False)
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    cax_a = fig.add_axes([0.135, 0.033, 0.20, 0.011])
    cb = fig.colorbar(im, cax=cax_a, orientation='horizontal')
    cb.set_label(r'Probe AUROC  ($\bigstar$ = best single view)', fontsize=6.0)
    cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_linewidth(0.4)
    ptitle(ax_a, 'a', 'Per-species probe AUROC by expert', x=-0.46, y=1.005)

    ax_b = fig.add_axes([0.575, 0.735, 0.40, 0.195])
    means = list(Av.mean(0)) + [Af.mean()]
    stds = list(Av.std(0)) + [Af.std()]
    xs = np.arange(6)
    ax_b.bar(xs, means, color=VC + [FUS], edgecolor='white', lw=0.4, width=0.68, zorder=2)
    ax_b.errorbar(xs, means, yerr=stds, fmt='none', ecolor='#2b2b2b',
                  elinewidth=0.8, capsize=2.6, capthick=0.8, zorder=3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax_b.text(i, m + s + 0.006, f'{m:.3f}', ha='center', va='bottom',
                  fontsize=5.6, fontweight='bold' if i == 5 else 'normal',
                  color=FUS if i == 5 else '#333333')
    ax_b.set_xticks(xs); ax_b.set_xticklabels(VS + ['Fus'], fontsize=6.0)
    ax_b.get_xticklabels()[5].set_color(FUS); ax_b.get_xticklabels()[5].set_fontweight('bold')
    ax_b.set_ylim(0.55, 1.04); ax_b.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax_b.set_ylabel('Linear-probe AUROC', fontsize=7.0)
    ax_b.spines[['top', 'right']].set_visible(False)
    ax_b.yaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax_b.set_axisbelow(True)
    ptitle(ax_b, 'b', 'Probe AUROC: experts vs. fusion', x=-0.135, y=1.03)

    ax_c = fig.add_axes([0.575, 0.375, 0.40, 0.205])
    o2 = np.argsort(accf)
    x = np.arange(23)
    ax_c.fill_between(x, accb[o2], acco[o2], color='#D9D9D9', alpha=0.7, lw=0,
                      zorder=1, label='Complementarity\nheadroom')
    ax_c.plot(x, acco[o2], color='#7f7f7f', lw=0.8, marker='^', ms=2.4, zorder=3,
              label=f'Oracle (any view)  {acco.mean():.3f}')
    ax_c.plot(x, accf[o2], color=FUS, lw=1.1, marker='o', ms=2.8, zorder=4,
              label=f'Fusion  {accf.mean():.3f}')
    ax_c.plot(x, accb[o2], color=VC[0], lw=0.9, marker='s', ms=2.4, zorder=3,
              label=f'Best single view  {accb.mean():.3f}')
    ax_c.set_xlabel('Species (sorted by fusion acc.)', fontsize=6.8)
    ax_c.set_ylabel('Accuracy', fontsize=7.0)
    ax_c.set_ylim(0.7, 1.0); ax_c.set_xlim(-0.5, 22.5)
    ax_c.spines[['top', 'right']].set_visible(False)
    ax_c.yaxis.grid(True, color='#EEEEEE', lw=0.6, zorder=0); ax_c.set_axisbelow(True)
    ax_c.legend(loc='lower right', fontsize=5.4, handlelength=1.3, labelspacing=0.3)
    ptitle(ax_c, 'c', 'Best-view, fusion and oracle accuracy', x=-0.135, y=1.03)

    ax_d = fig.add_axes([0.575, 0.075, 0.205, 0.185])
    im3 = ax_d.imshow(Dis, cmap='RdPu', vmin=0, vmax=Dis.max() * 1.05)
    ax_d.set_xticks(range(5)); ax_d.set_xticklabels(VS, rotation=40, ha='right', fontsize=5.6)
    ax_d.set_yticks(range(5)); ax_d.set_yticklabels(VS, fontsize=5.6)
    ax_d.tick_params(length=0)
    for a in range(5):
        for b in range(5):
            ax_d.text(b, a, f'{Dis[a,b]:.2f}', ha='center', va='center', fontsize=5.0,
                      color='white' if Dis[a, b] > Dis.max() * 0.62 else '#333333')
    ptitle(ax_d, 'd', 'Prediction disagreement', x=-0.34, y=1.04)

    ax_u = fig.add_axes([0.845, 0.075, 0.13, 0.185])
    yv = np.arange(5)[::-1]
    for i in range(5):
        ax_u.hlines(yv[i], 0, Uni[i] * 100, color=VC[i], lw=1.4, zorder=2)
        ax_u.plot(Uni[i] * 100, yv[i], 'o', ms=4.2, color=VC[i], zorder=3)
    ax_u.set_yticks(yv); ax_u.set_yticklabels(VS, fontsize=5.6)
    ax_u.set_xlim(0, max(Uni) * 100 * 1.35)
    ax_u.set_xlabel('Unique-correct (%)', fontsize=6.0)
    ax_u.tick_params(labelsize=5.2)
    ax_u.spines[['top', 'right']].set_visible(False)
    ax_u.set_title('Exclusive hits', fontsize=6.6, fontweight='bold', loc='left', x=-0.02, y=1.02)

    save_dir = os.path.join(HERE, 'panel')
    F.save_panels(fig, [
        ('a_winning_view_heatmap', ax_a, [cax_a]),
        ('b_probe_auroc_bars', ax_b, []),
        ('c_complementarity', ax_c, []),
        ('d_disagreement_unique', ax_d, [ax_u]),
    ], save_dir)

    out = os.path.join(HERE, 'fig_expert_complementarity')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', out + '.png')
    print(f'fusion AUROC {Af.mean():.4f} vs best single {max(Av.mean(0)):.4f}; '
          f'CKA off-diag {CKA[~np.eye(5,dtype=bool)].mean():.3f}; unique {Uni.round(4)}')

if __name__ == '__main__':
    main()
