import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import pearsonr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_V1 = os.path.join(_ROOT, 'figV1', '12.cross_species')
sys.path.insert(0, _V1)
from species_meta import SPECIES, PHYLUM_COLORS, short_name, ID2PHYLUM

DATA = os.path.join(_V1, 'cross_species_data')
MAT_DIR = os.path.join(_ROOT, 'iProV2', 'cross_species')
PARTS = os.path.join(_HERE, 'parts')
N = len(SPECIES)
IDS = [s[0] for s in SPECIES]
PHYLUM_ORDER = list(PHYLUM_COLORS)

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': '#333333',
    'axes.labelcolor': '#1a1a1a', 'text.color': '#1a1a1a',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.color': '#333333', 'ytick.color': '#333333', 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 150,
})
PANEL_KW = dict(fontsize=11, fontweight='bold', va='top', ha='left')

def load_tree(which='genome'):
    f = {'genome': ('genome_linkage.npy', 'genome_order.npy', 'genome_dist.npy'),
         'kmer': ('kmer_linkage.npy', 'kmer_order.npy', 'kmer_jsd.npy')}[which]
    return tuple(np.load(os.path.join(DATA, x)) for x in f)

def _rgb_col(colors):
    return np.array([mpl.colors.to_rgb(c) for c in colors]).reshape(-1, 1, 3)

def _rgb_row(colors):
    return np.array([mpl.colors.to_rgb(c) for c in colors]).reshape(1, -1, 3)

def phylum_strip(order):
    return [PHYLUM_COLORS[ID2PHYLUM[IDS[i]]] for i in order]

def save_bundle(fig, base):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')

def save_panels(fig, panels, out_dir, dpi=300, pad_in=0.06):
    os.makedirs(out_dir, exist_ok=True)
    all_axes = [a for _, main, subs in panels for a in [main, *subs]]
    for name, main, subs in panels:
        keep = {main, *subs}
        for a in [a for a in all_axes if a not in keep]:
            a.set_visible(False)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = Bbox.union([a.get_tightbbox(r) for a in (main, *subs)])
        ext = bb.padded(pad_in * fig.dpi).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=dpi, bbox_inches=ext)
        for a in [a for a in all_axes if a not in keep]:
            a.set_visible(True)
    fig.canvas.draw()

def clustermap_figure(M, Z, order, gc, outbase, cmap='viridis', ordered_by='Genome Mash'):
    Mo = M[np.ix_(order, order)]
    names = [short_name(SPECIES[i][1], IDS[i]) for i in order]
    ids_ord = [IDS[i] for i in order]
    ph_cols = phylum_strip(order)
    gc_ord = gc[order]
    gmin, gmax = float(np.min(gc)), float(np.max(gc))
    vmin, vmax = float(np.nanpercentile(Mo, 2)), float(np.nanmax(Mo))

    fig = plt.figure(figsize=(7.2, 7.0))
    L, B, W, H = 0.265, 0.085, 0.60, 0.60
    dd, tw, gap = 0.085, 0.018, 0.006
    ax_hm = fig.add_axes([L, B, W, H])
    ax_top = fig.add_axes([L, B + H + tw + 2 * gap, W, dd])
    ax_left = fig.add_axes([L - 2 * tw - 2 * gap - dd - gap, B, dd, H])
    ax_lph = fig.add_axes([L - tw, B, tw, H])
    ax_lgc = fig.add_axes([L - 2 * tw - gap, B, tw, H])
    ax_tph = fig.add_axes([L, B + H + gap, W, tw])
    ax_cb = fig.add_axes([L + W + 0.150, B + H * 0.42, 0.016, H * 0.40])
    ax_gccb = fig.add_axes([L + W + 0.150, B + H * 0.06, 0.016, H * 0.22])

    Ze = Z.copy(); Ze[:, 2] = np.arange(1, len(Ze) + 1)
    dendrogram(Ze, ax=ax_top, orientation='top', no_labels=True, link_color_func=lambda k: '#555555')
    dendrogram(Ze, ax=ax_left, orientation='left', no_labels=True, link_color_func=lambda k: '#555555')
    ax_left.invert_yaxis()
    for ax in (ax_top, ax_left):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    im = ax_hm.imshow(Mo, aspect='equal', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    for k in range(N):
        ax_hm.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor='#D81B60', lw=0.9, zorder=5))
    ax_hm.set_xticks(range(N)); ax_hm.set_xticklabels(ids_ord, fontsize=5.6)
    ax_hm.set_yticks(range(N)); ax_hm.yaxis.tick_right()
    ax_hm.set_yticklabels(names, fontsize=5.8, fontstyle='italic')
    ax_hm.tick_params(length=0); ax_hm.set_xlabel('Tested on (species id)', fontsize=8)
    fig.text(L - 2 * tw - 2 * gap - dd - gap - 0.020, B + H / 2, 'Model trained on',
             rotation=90, va='center', ha='center', fontsize=8)
    for sp in ax_hm.spines.values():
        sp.set_edgecolor('#888888')

    ax_lph.imshow(_rgb_col(ph_cols), aspect='auto', origin='upper')
    ax_lgc.imshow(gc_ord.reshape(-1, 1), aspect='auto', origin='upper', cmap='cividis', vmin=gmin, vmax=gmax)
    ax_tph.imshow(_rgb_row(ph_cols), aspect='auto', origin='upper')
    for ax, lab in [(ax_lph, 'Phylum'), (ax_lgc, 'GC')]:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=5.6, rotation=90, labelpad=1.5)
    ax_tph.set_xticks([]); ax_tph.set_yticks([])

    cb = fig.colorbar(im, cax=ax_cb); cb.set_label('Transfer MCC', fontsize=8); cb.ax.tick_params(labelsize=6)
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(gmin, gmax), cmap='cividis'), cax=ax_gccb)
    cb2.set_label('GC content', fontsize=7); cb2.ax.tick_params(labelsize=6)
    cb2.set_ticks([round(gmin, 2), round(gmax, 2)])

    handles = [Rectangle((0, 0), 1, 1, color=PHYLUM_COLORS[p]) for p in PHYLUM_ORDER]
    fig.legend(handles, PHYLUM_ORDER, loc='lower left', bbox_to_anchor=(0.015, 0.015),
               fontsize=5.8, frameon=False, title='Phylum', title_fontsize=6.2,
               handlelength=1.0, labelspacing=0.32)

    tree_tag = {'Genome Mash': 'genome-tree ordered', 'Promoter k-mer': 'promoter k-mer-tree ordered'}
    ty = B + H + 2 * tw + 3 * gap + dd
    fig.text(L - 2 * tw - 2 * gap - dd - gap, ty + 0.012, 'Cross-species promoter transfer (MCC)',
             fontsize=9.5, fontweight='bold')
    fig.text(L - 2 * tw - 2 * gap - dd - gap, ty - 0.006,
             'DeepPro-v2 · non-zero-shot (shared TAPT backbones) · '
             + tree_tag.get(ordered_by, ordered_by) + ' · red box = self-test',
             fontsize=6.0, color='#555555')
    save_bundle(fig, outbase); plt.close(fig)
    print(f'saved {outbase}.{{png,pdf}}  [{ordered_by} order]')

def tanglegram_figure(Z1, Z2, outbase, name1='Genome Mash', name2='Promoter k-mer'):
    fig = plt.figure(figsize=(6.6, 5.6))
    axL = fig.add_axes([0.02, 0.07, 0.30, 0.83])
    axM = fig.add_axes([0.40, 0.07, 0.20, 0.83])
    axR = fig.add_axes([0.68, 0.07, 0.30, 0.83])

    d1 = dendrogram(Z1, orientation='left', ax=axL, no_labels=True, link_color_func=lambda k: '#999999')
    d2 = dendrogram(Z2, orientation='right', ax=axR, no_labels=True, link_color_func=lambda k: '#999999')
    for ax in (axL, axM, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axL.invert_xaxis()

    leaves1, leaves2 = d1['leaves'], d2['leaves']
    pos1 = {orig: i for i, orig in enumerate(leaves1)}
    pos2 = {orig: i for i, orig in enumerate(leaves2)}
    yr = (0, 10 * N)
    for ax in (axL, axM, axR):
        ax.set_ylim(*yr)
    axM.set_xlim(0, 1)

    for orig in range(N):
        y1 = 10 * pos1[orig] + 5
        y2 = 10 * pos2[orig] + 5
        col = PHYLUM_COLORS[ID2PHYLUM[IDS[orig]]]
        axM.plot([0, 1], [y1, y2], color=col, lw=1.0, alpha=0.85)
        axM.text(-0.04, y1, str(IDS[orig]), ha='right', va='center', fontsize=5.6)
        axM.text(1.04, y2, str(IDS[orig]), ha='left', va='center', fontsize=5.6)

    axL.set_title(name1 + ' tree', fontsize=8, fontweight='bold')
    axR.set_title(name2 + ' tree', fontsize=8, fontweight='bold')
    fig.text(0.5, 0.965, 'Genome phylogeny vs promoter composition (tanglegram)',
             ha='center', fontsize=9, fontweight='bold')
    handles = [Line2D([0], [0], color=PHYLUM_COLORS[p], lw=2) for p in PHYLUM_ORDER]
    fig.legend(handles, PHYLUM_ORDER, loc='lower center', ncol=3,
               fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    save_bundle(fig, outbase); plt.close(fig)
    print(f'saved {outbase}.{{png,pdf}}  [tanglegram]')

def mantel(D1, D2, perms=9999, seed=0):
    iu = np.triu_indices_from(D1, k=1)
    x, y = D1[iu], D2[iu]
    r = pearsonr(x, y)[0]
    rng = np.random.default_rng(seed)
    n = D1.shape[0]; cnt = 0
    for _ in range(perms):
        p = rng.permutation(n)
        if abs(pearsonr(x, D2[np.ix_(p, p)][iu])[0]) >= abs(r):
            cnt += 1
    return r, (cnt + 1) / (perms + 1)

def _scatter(ax, M, D, dist_label, letter, title_text, show_legend=False):
    xs, ys, same = [], [], []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            xs.append(D[a, b]); ys.append(M[a, b])
            same.append(ID2PHYLUM[IDS[a]] == ID2PHYLUM[IDS[b]])
    xs, ys, same = np.array(xs), np.array(ys), np.array(same)
    ax.scatter(xs[~same], ys[~same], s=9, c='#C2C2C2', edgecolor='none', alpha=0.75,
               label='Different phylum', zorder=2)
    ax.scatter(xs[same], ys[same], s=15, c='#C1272D', edgecolor='white', linewidth=0.25,
               alpha=0.9, label='Same phylum', zorder=3)
    k, b0 = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, k * xx + b0, color='#1a1a1a', lw=1.2, zorder=4)
    r, p = mantel(D, (M + M.T) / 2)
    ax.text(0.96, 0.96, f'Mantel $r$ = {r:.2f}\n$P$ = {p:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=6.8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D9D9D9', lw=0.5))
    ax.set_xlabel(f'{dist_label} distance', fontsize=7.8)
    ax.set_ylabel('Transfer MCC (model i → species j)', fontsize=7.6)
    ax.set_ylim(min(0, ys.min()) - 0.03, 1.0)
    ax.spines[['top', 'right']].set_visible(False)
    if show_legend:
        ax.legend(fontsize=6.0, frameon=False, loc='lower left')
    ax.text(-0.20, 1.09, letter, transform=ax.transAxes, **PANEL_KW)
    ax.text(0.0, 1.02, title_text, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    return r, p

def analysis_figure(M, dists, order, outbase):
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))
    axa, axb, axc, axd = axes
    items = list(dists.items())
    rs = {}
    rs['genome'] = _scatter(axa, M, items[0][1], items[0][0], 'a', 'Transfer vs genome phylogeny', True)
    rs['kmer'] = _scatter(axb, M, items[1][1], items[1][0], 'b', 'Transfer vs promoter composition')

    Ao = (M - M.T)[np.ix_(order, order)]
    ids_ord = [IDS[i] for i in order]
    vmax = np.nanmax(np.abs(Ao))
    im = axc.imshow(Ao, cmap='RdBu_r', origin='upper',
                    norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect='equal')
    axc.set_xticks(range(N)); axc.set_xticklabels(ids_ord, fontsize=4.6)
    axc.set_yticks(range(N)); axc.set_yticklabels(ids_ord, fontsize=4.6)
    axc.tick_params(length=0)
    axc.set_xlabel('Tested on (id)', fontsize=7.6); axc.set_ylabel('Trained on (id)', fontsize=7.6)
    cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04)
    cb.set_label('MCC(i→j) − MCC(j→i)', fontsize=6.6); cb.ax.tick_params(labelsize=6)
    axc.text(-0.20, 1.09, 'c', transform=axc.transAxes, **PANEL_KW)
    axc.text(0.0, 1.02, 'Directional transfer asymmetry', transform=axc.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    self_mcc = np.array([M[j, j] for j in range(N)])
    best_ext = np.array([max(M[i, j] for i in range(N) if i != j) for j in range(N)])
    ph_cols = [PHYLUM_COLORS[ID2PHYLUM[IDS[j]]] for j in range(N)]
    axd.plot([0, 1], [0, 1], ls='--', color='#999999', lw=0.8, zorder=1)
    axd.scatter(self_mcc, best_ext, s=26, c=ph_cols, edgecolor='white', linewidth=0.4, zorder=3)
    lo = min(self_mcc.min(), best_ext.min()) - 0.03
    axd.set_xlim(lo, 1.0); axd.set_ylim(lo, 1.0)
    axd.set_xlabel('Self-trained MCC', fontsize=7.8)
    axd.set_ylabel('Best cross-species (external) MCC', fontsize=7.4)
    gap_mean = float((self_mcc - best_ext).mean())
    axd.text(0.96, 0.06, f'mean gap = {gap_mean:+.3f}\n(self − best external)', transform=axd.transAxes,
             ha='right', va='bottom', fontsize=6.4,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D9D9D9', lw=0.5))
    axd.spines[['top', 'right']].set_visible(False)
    axd.text(-0.20, 1.09, 'd', transform=axd.transAxes, **PANEL_KW)
    axd.text(0.0, 1.02, 'Generalization boundary', transform=axd.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    fig.subplots_adjust(left=0.05, right=0.985, top=0.88, bottom=0.14, wspace=0.42)
    save_panels(fig, [
        ('a_transfer_vs_genome', axa, []),
        ('b_transfer_vs_promoter', axb, []),
        ('c_directional_asymmetry', axc, [cb.ax]),
        ('d_generalization_boundary', axd, []),
    ], PARTS)
    save_bundle(fig, outbase); plt.close(fig)
    print('saved %s | Mantel ' % outbase + ', '.join(f'{k}: r={r:.3f} p={p:.4f}' for k, (r, p) in rs.items())
          + f' | self-MCC mean={self_mcc.mean():.3f} best-ext mean={best_ext.mean():.3f}')

def main():
    Zg, og, Dg = load_tree('genome')
    Zk, ok, Dk = load_tree('kmer')
    M = np.load(os.path.join(MAT_DIR, 'mcc_matrix.npy'))
    gc = np.load(os.path.join(DATA, 'gc.npy'))
    print(f'V2 transfer matrix: diag mean={np.nanmean(np.diag(M)):.4f} '
          f'off-diag mean={np.nanmean(M[~np.eye(N, dtype=bool)]):.4f}')
    clustermap_figure(M, Zg, og, gc, os.path.join(PARTS, 'fig_mcc_genome_tree'), ordered_by='Genome Mash')
    clustermap_figure(M, Zk, ok, gc, os.path.join(PARTS, 'fig_mcc_kmer_tree'), ordered_by='Promoter k-mer')
    tanglegram_figure(Zg, Zk, os.path.join(PARTS, 'fig_tanglegram'))
    analysis_figure(M, {'Genome Mash': Dg, 'Promoter k-mer': Dk}, og,
                    os.path.join(PARTS, 'fig_analysis'))

if __name__ == '__main__':
    main()
