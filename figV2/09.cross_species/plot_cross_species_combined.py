import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
from scipy.cluster.hierarchy import dendrogram

import plot_cross_species as P
from species_meta import short_name

_HERE = P._HERE
N, IDS, SPECIES = P.N, P.IDS, P.SPECIES
ID2PHYLUM, PHYLUM_COLORS, PHYLUM_ORDER = P.ID2PHYLUM, P.PHYLUM_COLORS, P.PHYLUM_ORDER
PANEL_KW = dict(fontsize=15, fontweight='bold', va='top', ha='left')
TREE_TAG = {'Genome Mash': 'genome Mash tree', 'Promoter k-mer': 'promoter k-mer tree'}

def draw_clustermap(sf, M, Z, order, gc, letter, ordered_by):
    Mo = M[np.ix_(order, order)]
    names = [short_name(SPECIES[i][1], IDS[i]) for i in order]
    ids_ord = [IDS[i] for i in order]
    ph_cols = P.phylum_strip(order)
    gc_ord = gc[order]
    gmin, gmax = float(np.min(gc)), float(np.max(gc))
    vmin, vmax = float(np.nanpercentile(Mo, 2)), float(np.nanmax(Mo))

    L, B, W, H = 0.165, 0.105, 0.550, 0.550
    dd, tw, gap = 0.075, 0.016, 0.006
    ax_hm = sf.add_axes([L, B, W, H])
    ax_top = sf.add_axes([L, B + H + tw + 2 * gap, W, dd])
    ax_left = sf.add_axes([L - 2 * tw - 2 * gap - dd - gap, B, dd, H])
    ax_lph = sf.add_axes([L - tw, B, tw, H])
    ax_lgc = sf.add_axes([L - 2 * tw - gap, B, tw, H])
    ax_tph = sf.add_axes([L, B + H + gap, W, tw])
    ax_cb = sf.add_axes([L + W + 0.150, B + H * 0.42, 0.016, H * 0.40])
    ax_gccb = sf.add_axes([L + W + 0.150, B + H * 0.06, 0.016, H * 0.22])

    Ze = Z.copy(); Ze[:, 2] = np.arange(1, len(Ze) + 1)
    dendrogram(Ze, ax=ax_top, orientation='top', no_labels=True, link_color_func=lambda k: '#555555')
    dendrogram(Ze, ax=ax_left, orientation='left', no_labels=True, link_color_func=lambda k: '#555555')
    ax_left.invert_yaxis()
    for ax in (ax_top, ax_left):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    im = ax_hm.imshow(Mo, aspect='equal', origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
    for k in range(N):
        ax_hm.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor='#D81B60', lw=0.9, zorder=5))
    ax_hm.set_xticks(range(N)); ax_hm.set_xticklabels(ids_ord, fontsize=6.6)
    ax_hm.set_yticks(range(N)); ax_hm.yaxis.tick_right()
    ax_hm.set_yticklabels(names, fontsize=6.8, fontstyle='italic')
    ax_hm.tick_params(length=0); ax_hm.set_xlabel('Tested on (species id)', fontsize=9)
    sf.text(L - 2 * tw - 2 * gap - dd - gap - 0.022, B + H / 2, 'Model trained on',
            rotation=90, va='center', ha='center', fontsize=9)
    for sp in ax_hm.spines.values():
        sp.set_edgecolor('#888888')

    ax_lph.imshow(P._rgb_col(ph_cols), aspect='auto', origin='upper')
    ax_lgc.imshow(gc_ord.reshape(-1, 1), aspect='auto', origin='upper', cmap='cividis', vmin=gmin, vmax=gmax)
    ax_tph.imshow(P._rgb_row(ph_cols), aspect='auto', origin='upper')
    for ax, lab in [(ax_lph, 'Phylum'), (ax_lgc, 'GC')]:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=6.6, rotation=90, labelpad=1.5)
    ax_tph.set_xticks([]); ax_tph.set_yticks([])

    cb = sf.colorbar(im, cax=ax_cb); cb.set_label('Transfer MCC', fontsize=9); cb.ax.tick_params(labelsize=7)
    cb2 = sf.colorbar(ScalarMappable(norm=Normalize(gmin, gmax), cmap='cividis'), cax=ax_gccb)
    cb2.set_label('GC content', fontsize=7.5); cb2.ax.tick_params(labelsize=7)
    cb2.set_ticks([round(gmin, 2), round(gmax, 2)])

    ytop = B + H + 2 * tw + 3 * gap + dd
    lx = L - 2 * tw - 2 * gap - dd - gap
    sf.text(lx, ytop + 0.058, letter, fontsize=15, fontweight='bold', va='bottom', ha='left')
    sf.text(lx + 0.050, ytop + 0.070, 'Cross-species transfer (MCC)', fontsize=11.5, fontweight='bold', va='bottom')
    sf.text(lx + 0.050, ytop + 0.050,
            'DeepPro-v2 · non-zero-shot (shared TAPT backbones) · '
            + TREE_TAG[ordered_by] + ' ordered · red box = self-test',
            fontsize=7.0, color='#555555', va='bottom')

def draw_tanglegram(sf, Z1, Z2, letter, name1='Genome Mash', name2='Promoter k-mer'):
    axL = sf.add_axes([0.05, 0.08, 0.27, 0.78])
    axM = sf.add_axes([0.40, 0.08, 0.20, 0.78])
    axR = sf.add_axes([0.68, 0.08, 0.27, 0.78])

    d1 = dendrogram(Z1, orientation='left', ax=axL, no_labels=True, link_color_func=lambda k: '#999999')
    d2 = dendrogram(Z2, orientation='right', ax=axR, no_labels=True, link_color_func=lambda k: '#999999')
    for ax in (axL, axM, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axL.invert_xaxis()

    pos1 = {orig: i for i, orig in enumerate(d1['leaves'])}
    pos2 = {orig: i for i, orig in enumerate(d2['leaves'])}
    for ax in (axL, axM, axR):
        ax.set_ylim(0, 10 * N)
    axM.set_xlim(0, 1)
    for orig in range(N):
        y1, y2 = 10 * pos1[orig] + 5, 10 * pos2[orig] + 5
        col = PHYLUM_COLORS[ID2PHYLUM[IDS[orig]]]
        axM.plot([0, 1], [y1, y2], color=col, lw=1.1, alpha=0.85)
        axM.text(-0.05, y1, str(IDS[orig]), ha='right', va='center', fontsize=6.6)
        axM.text(1.05, y2, str(IDS[orig]), ha='left', va='center', fontsize=6.6)

    axL.set_title(name1 + '\ntree', fontsize=8.5, fontweight='bold')
    axR.set_title(name2 + '\ntree', fontsize=8.5, fontweight='bold')
    sf.text(0.02, 0.99, letter, fontsize=15, fontweight='bold', va='top', ha='left')
    sf.text(0.5, 0.975, 'Genome phylogeny vs promoter composition',
            ha='center', va='top', fontsize=10.5, fontweight='bold')

def draw_scatter(sf, M, D, dist_label, letter, title_text, show_legend=False):
    ax = sf.add_axes([0.20, 0.15, 0.74, 0.72])
    xs, ys, same = [], [], []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            xs.append(D[a, b]); ys.append(M[a, b])
            same.append(ID2PHYLUM[IDS[a]] == ID2PHYLUM[IDS[b]])
    xs, ys, same = np.array(xs), np.array(ys), np.array(same)
    ax.scatter(xs[~same], ys[~same], s=11, c='#C2C2C2', edgecolor='none', alpha=0.75,
               label='Different phylum', zorder=2)
    ax.scatter(xs[same], ys[same], s=18, c='#C1272D', edgecolor='white', linewidth=0.25,
               alpha=0.9, label='Same phylum', zorder=3)
    k, b0 = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, k * xx + b0, color='#1a1a1a', lw=1.3, zorder=4)
    r, p = P.mantel(D, (M + M.T) / 2)
    ax.text(0.96, 0.96, f'Mantel $r$ = {r:.2f}\n$P$ = {p:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D9D9D9', lw=0.5))
    ax.set_xlabel(f'{dist_label} distance', fontsize=10)
    ax.set_ylabel('Transfer MCC (model i → species j)', fontsize=9.6)
    ax.set_ylim(min(0, ys.min()) - 0.03, 1.0)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    if show_legend:
        ax.legend(fontsize=8.0, frameon=False, loc='lower left')
    ax.text(-0.22, 1.11, letter, transform=ax.transAxes, **PANEL_KW)
    ax.text(0.0, 1.03, title_text, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
    return r, p

def draw_asymmetry(sf, M, order, letter):
    ax = sf.add_axes([0.17, 0.14, 0.64, 0.74])
    Ao = (M - M.T)[np.ix_(order, order)]
    ids_ord = [IDS[i] for i in order]
    vmax = np.nanmax(np.abs(Ao))
    im = ax.imshow(Ao, cmap='RdBu_r', origin='upper',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect='equal')
    ax.set_xticks(range(N)); ax.set_xticklabels(ids_ord, fontsize=7.0)
    ax.set_yticks(range(N)); ax.set_yticklabels(ids_ord, fontsize=7.0)
    ax.tick_params(length=0)
    ax.set_xlabel('Tested on (id)', fontsize=10); ax.set_ylabel('Model trained on (id)', fontsize=10)
    cb = sf.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('MCC(i→j) − MCC(j→i)', fontsize=8.5); cb.ax.tick_params(labelsize=8)
    ax.text(-0.19, 1.07, letter, transform=ax.transAxes, **PANEL_KW)
    ax.text(0.0, 1.02, 'Directional transfer asymmetry', transform=ax.transAxes,
            fontsize=10.5, fontweight='bold', va='bottom')

def draw_boundary(sf, M, letter):
    ax = sf.add_axes([0.15, 0.14, 0.78, 0.74])
    self_mcc = np.array([M[j, j] for j in range(N)])
    best_ext = np.array([max(M[i, j] for i in range(N) if i != j) for j in range(N)])
    ph_cols = [PHYLUM_COLORS[ID2PHYLUM[IDS[j]]] for j in range(N)]
    ax.plot([0, 1], [0, 1], ls='--', color='#999999', lw=0.8, zorder=1)
    ax.scatter(self_mcc, best_ext, s=34, c=ph_cols, edgecolor='white', linewidth=0.4, zorder=3)
    lo = min(self_mcc.min(), best_ext.min()) - 0.03
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
    ax.set_xlabel('Self-trained MCC', fontsize=10)
    ax.set_ylabel('Best cross-species (external) MCC', fontsize=9.6)
    gap_mean = float((self_mcc - best_ext).mean())
    ax.text(0.96, 0.06, f'mean gap = {gap_mean:+.3f}\n(self − best external)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.0,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D9D9D9', lw=0.5))
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.17, 1.07, letter, transform=ax.transAxes, **PANEL_KW)
    ax.text(0.0, 1.02, 'Generalization boundary', transform=ax.transAxes,
            fontsize=10.5, fontweight='bold', va='bottom')

def _phylum_legend(sf, title):
    handles = [Rectangle((0, 0), 1, 1, color=PHYLUM_COLORS[p]) for p in PHYLUM_ORDER]
    sf.legend(handles, PHYLUM_ORDER, loc='center', ncol=len(PHYLUM_ORDER),
              fontsize=9, frameon=False, title=title,
              title_fontsize=9.5, handlelength=1.2, columnspacing=1.8)

def save_bundle(fig, base):
    fig.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')

def build_heatmaps(M, Zg, og, Zk, ok, gc):
    fig = plt.figure(figsize=(14.0, 7.4))
    sf_hm, sf_leg = fig.subfigures(2, 1, height_ratios=[7.0, 0.4])
    grid = sf_hm.subfigures(1, 2, wspace=0.0)
    draw_clustermap(grid[0], M, Zg, og, gc, 'a', 'Genome Mash')
    draw_clustermap(grid[1], M, Zk, ok, gc, 'b', 'Promoter k-mer')
    _phylum_legend(sf_leg, 'Phylum (heatmap tracks)')

    base = os.path.join(_HERE, 'fig_cross_species_heatmaps')
    save_bundle(fig, base)
    fig.canvas.draw()
    for name, sub in {'a_genome_heatmap': grid[0], 'b_kmer_heatmap': grid[1]}.items():
        bb = sub.bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(P.PARTS, f'combined_{name}.png'), dpi=300,
                    bbox_inches=Bbox(bb.get_points()).padded(0.04))
    plt.close(fig)
    print(f'saved {base}.{{png,pdf}}')

def build_analysis(M, Zg, Zk, Dg, Dk, og):
    fig = plt.figure(figsize=(14.0, 9.4))
    sf_top, sf_leg, sf_bot = fig.subfigures(3, 1, height_ratios=[4.7, 0.4, 4.3])
    grid_top = sf_top.subfigures(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.0)
    grid_bot = sf_bot.subfigures(1, 2, width_ratios=[1.12, 1.0], wspace=0.0)

    draw_tanglegram(grid_top[0], Zg, Zk, 'c')
    rg = draw_scatter(grid_top[1], M, Dg, 'Genome Mash', 'd', 'Transfer vs genome phylogeny', show_legend=True)
    rk = draw_scatter(grid_top[2], M, Dk, 'Promoter k-mer', 'e', 'Transfer vs promoter composition')
    draw_asymmetry(grid_bot[0], M, og, 'f')
    draw_boundary(grid_bot[1], M, 'g')
    _phylum_legend(sf_leg, 'Phylum (tanglegram links · scatter · boundary points)')

    base = os.path.join(_HERE, 'fig_cross_species_analysis')
    save_bundle(fig, base)
    fig.canvas.draw()
    sub_map = {'c_tanglegram': grid_top[0], 'd_mantel_genome': grid_top[1],
               'e_mantel_kmer': grid_top[2], 'f_asymmetry': grid_bot[0], 'g_boundary': grid_bot[1]}
    for name, sub in sub_map.items():
        bb = sub.bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(P.PARTS, f'combined_{name}.png'), dpi=300,
                    bbox_inches=Bbox(bb.get_points()).padded(0.04))
    plt.close(fig)
    print(f'saved {base}.{{png,pdf}} | Mantel genome r={rg[0]:.3f} p={rg[1]:.4f} · '
          f'kmer r={rk[0]:.3f} p={rk[1]:.4f}')

def main():
    Zg, og, Dg = P.load_tree('genome')
    Zk, ok, Dk = P.load_tree('kmer')
    M = np.load(os.path.join(P.MAT_DIR, 'mcc_matrix.npy'))
    gc = np.load(os.path.join(P.DATA, 'gc.npy'))
    os.makedirs(P.PARTS, exist_ok=True)
    print(f'V2 transfer matrix: diag mean={np.nanmean(np.diag(M)):.4f} '
          f'off-diag mean={np.nanmean(M[~np.eye(N, dtype=bool)]):.4f}')
    build_heatmaps(M, Zg, og, Zk, ok, gc)
    build_analysis(M, Zg, Zk, Dg, Dk, og)

if __name__ == '__main__':
    main()
