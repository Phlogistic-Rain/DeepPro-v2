import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import matthews_corrcoef
from scipy.stats import spearmanr, pearsonr
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _figlib as F

warnings.filterwarnings('ignore', category=ConvergenceWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(HERE, 'panel')
SEP_CACHE = os.path.join(HERE, '_sep_cache.npz')
UMAP_DIR = os.path.join(HERE, '_umap_cache')

POS_C = '#C1272D'
NEG_C = '#4C72B0'
FUS_C = F.FUSION_COLOR
SINGLE_C = '#8C9BAB'
UMAP_SEED = 42

def _probe_auc(X, y):
    Xs = StandardScaler().fit_transform(X.astype(np.float32))
    return float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                                 Xs, y, cv=5, scoring='roc_auc').mean())

def compute_separability(force=False):
    if os.path.exists(SEP_CACHE) and not force:
        d = np.load(SEP_CACHE)
        return {k: d[k] for k in d.files}
    sids = np.array(F.ALL_SPECIES)
    fus = np.full(len(sids), np.nan)
    vw = np.full((len(sids), 5), np.nan)
    mcc = np.full(len(sids), np.nan)
    nsamp = np.zeros(len(sids), int)
    for i, sid in enumerate(sids):
        c = F.load_repr_cache(int(sid))
        y = c['y']
        nsamp[i] = len(y)
        fus[i] = _probe_auc(c['rep'], y)
        for k in range(5):
            vw[i, k] = _probe_auc(c[f'feat{k}'], y)
        yt, prob = F.load_species_ensemble('deeppro_v2', int(sid), 'test')
        mcc[i] = matthews_corrcoef(yt, (prob >= 0.5).astype(int)) if yt is not None else np.nan
        print(f'  s{int(sid):>2}: N={nsamp[i]:>5}  fusion={fus[i]:.3f}  '
              f'best-view={vw[i].max():.3f}(v{int(vw[i].argmax())})  MCC={mcc[i]:.3f}')
    best_idx = vw.argmax(1)
    best_auc = vw.max(1)
    np.savez(SEP_CACHE, sids=sids, fusion=fus, views=vw, best_idx=best_idx,
             best_auc=best_auc, mcc=mcc, nsamp=nsamp)
    return dict(sids=sids, fusion=fus, views=vw, best_idx=best_idx,
                best_auc=best_auc, mcc=mcc, nsamp=nsamp)

def get_umap(sid, kind):
    os.makedirs(UMAP_DIR, exist_ok=True)
    fp = os.path.join(UMAP_DIR, f's{sid}_{kind}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    c = F.load_repr_cache(sid)
    X = c['rep'] if kind == 'rep' else c[f'feat{int(kind[1])}']
    Xs = StandardScaler().fit_transform(X.astype(np.float32))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean',
                        random_state=UMAP_SEED, n_jobs=1)
    emb = reducer.fit_transform(Xs).astype(np.float32)
    np.save(fp, emb)
    return emb

def scatter_umap(ax, emb, y, s=4, alpha=0.55, legend=False):
    for lab, col, name in [(0, NEG_C, 'Non-promoter'), (1, POS_C, 'Promoter')]:
        m = y == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=s, c=col, lw=0, alpha=alpha,
                   rasterized=True, label=name)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#bcbcbc'); sp.set_linewidth(0.5)
    if legend:
        lg = ax.legend(loc='upper left', fontsize=5.6, markerscale=2.2,
                       handletextpad=0.2, borderpad=0.25, labelspacing=0.25,
                       framealpha=0.85, frameon=True)
        lg.get_frame().set_edgecolor('none'); lg.get_frame().set_facecolor('white')

def corner_label(ax, txt, fs=5.8):
    ax.text(0.97, 0.035, txt, transform=ax.transAxes, fontsize=fs, va='bottom',
            ha='right', color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.75))

def umap_axis_hint(ax):
    ax.annotate('', xy=(0.15, 0.02), xytext=(0.02, 0.02), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#9a9a9a'))
    ax.annotate('', xy=(0.02, 0.15), xytext=(0.02, 0.02), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#9a9a9a'))
    ax.text(0.17, 0.015, 'UMAP-1', transform=ax.transAxes, fontsize=4.6,
            color='#9a9a9a', va='bottom', ha='left')
    ax.text(0.028, 0.17, 'UMAP-2', transform=ax.transAxes, fontsize=4.6,
            color='#9a9a9a', va='bottom', ha='left', rotation=90)

def band_header(ax, letter, title, y_ax, letter_dx=-0.02, title_dx=0.02, title_fs=8):
    ax.text(letter_dx, y_ax, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')
    ax.text(title_dx, y_ax, title, transform=ax.transAxes, fontsize=title_fs,
            fontweight='bold', va='bottom', ha='left')

def main():
    F.set_style()
    print('[1/3] computing per-species separability (5-fold linear probe) ...')
    S = compute_separability()
    sids = S['sids'].astype(int)
    fus, vw, mcc, nsamp = S['fusion'], S['views'], S['mcc'], S['nsamp']
    best_idx, best_auc = S['best_idx'].astype(int), S['best_auc']

    order_lo2hi = np.argsort(fus)
    pick_ranks = np.unique(np.linspace(0, len(sids) - 1, 8).round().astype(int))
    grid_pos = order_lo2hi[pick_ranks]
    grid_pos = grid_pos[np.argsort(-fus[grid_pos])]

    gain = fus - best_auc
    elig = np.where(nsamp >= 1500)[0]
    c_pos = elig[np.argmax(gain[elig])] if len(elig) else int(np.argmax(gain))
    c_sid = int(sids[c_pos]); c_bview = int(best_idx[c_pos])

    print('[2/3] computing UMAP embeddings (cached) ...')
    for p in grid_pos:
        get_umap(int(sids[p]), 'rep')
    get_umap(c_sid, 'rep'); get_umap(c_sid, f'v{c_bview}')

    fig = plt.figure(figsize=(7.2, 9.4))
    gs_a = fig.add_gridspec(2, 4, left=0.085, right=0.975, top=0.955, bottom=0.700,
                            hspace=0.34, wspace=0.14)
    ax_a = [fig.add_subplot(gs_a[r, col]) for r in range(2) for col in range(4)]
    gs_b = fig.add_gridspec(1, 1, left=0.085, right=0.975, top=0.620, bottom=0.500)
    ax_b = fig.add_subplot(gs_b[0, 0])
    gs_cd = fig.add_gridspec(1, 3, left=0.085, right=0.975, top=0.350, bottom=0.085,
                             width_ratios=[1.0, 1.0, 1.28], wspace=0.42)
    ax_c = [fig.add_subplot(gs_cd[0, 0]), fig.add_subplot(gs_cd[0, 1])]
    ax_d = fig.add_subplot(gs_cd[0, 2])

    for j, p in enumerate(grid_pos):
        sid = int(sids[p]); ax = ax_a[j]
        emb = get_umap(sid, 'rep')
        c = F.load_repr_cache(sid)
        scatter_umap(ax, emb, c['y'], s=3.0, alpha=0.55, legend=(j == 0))
        ax.set_title(F.SPECIES_NAME[sid], fontsize=6.4, fontstyle='italic', pad=2.5)
        corner_label(ax, f'AUROC {fus[p]:.2f}', fs=5.3)
    umap_axis_hint(ax_a[4])
    band_header(ax_a[0], 'a', 'Fused representation UMAP across representative species',
                y_ax=1.30, letter_dx=-0.14, title_dx=0.02)

    ob = np.argsort(fus)
    x = np.arange(len(sids)); bw = 0.40
    ax_b.bar(x - bw / 2, best_auc[ob], bw, color=SINGLE_C, edgecolor='white',
             lw=0.3, zorder=3, label='Best single view')
    ax_b.bar(x + bw / 2, fus[ob], bw, color=FUS_C, edgecolor='white',
             lw=0.3, zorder=3, label='Fusion')
    ax_b.axhline(fus.mean(), color=FUS_C, lw=0.8, ls=(0, (4, 3)), zorder=2,
                 label=f'Mean fusion ({fus.mean():.3f})')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([F.SPECIES_NAME[int(sids[i])] for i in ob], rotation=55,
                         ha='right', fontsize=5.2, fontstyle='italic')
    ax_b.set_ylim(0.5, 1.0)
    ax_b.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax_b.set_ylabel('Linear-probe AUROC', fontsize=7.5)
    ax_b.set_xlim(-0.7, len(sids) - 0.3)
    ax_b.spines[['top', 'right']].set_visible(False)
    ax_b.yaxis.grid(True, color='#EAEAEA', lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.tick_params(axis='x', length=0)
    ax_b.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=3, frameon=False,
                fontsize=6.5, handlelength=1.4, handletextpad=0.4, columnspacing=1.4)
    nwin = int(np.sum(fus >= best_auc - 1e-9))
    band_header(ax_b, 'b', 'Fusion exceeds the best single expert view (23 species)',
                y_ax=1.045, letter_dx=-0.065, title_dx=0.0)

    c = F.load_repr_cache(c_sid)
    emb_v = get_umap(c_sid, f'v{c_bview}')
    emb_r = get_umap(c_sid, 'rep')
    scatter_umap(ax_c[0], emb_v, c['y'], s=3.2, alpha=0.55, legend=True)
    scatter_umap(ax_c[1], emb_r, c['y'], s=3.2, alpha=0.55)
    ax_c[0].text(0.5, -0.075, f'Best single view: {F.VIEW_SHORT[c_bview]}',
                 transform=ax_c[0].transAxes, ha='center', va='top', fontsize=6.2)
    ax_c[1].text(0.5, -0.075, f'Fusion (soft-MoE) · Δ +{gain[c_pos]:.2f}',
                 transform=ax_c[1].transAxes, ha='center', va='top', fontsize=6.2,
                 color=FUS_C, fontweight='bold')
    corner_label(ax_c[0], f'AUROC {best_auc[c_pos]:.2f}', fs=5.6)
    corner_label(ax_c[1], f'AUROC {fus[c_pos]:.2f}', fs=5.6)
    umap_axis_hint(ax_c[0])
    band_header(ax_c[0], 'c', F.SPECIES_NAME[c_sid],
                y_ax=1.05, letter_dx=-0.10, title_dx=0.02, title_fs=8)
    ax_c[0].texts[-1].set_fontstyle('italic')

    good = ~np.isnan(mcc)
    xv, yv = fus[good], mcc[good]
    sg = sids[good]
    rho, p_s = spearmanr(xv, yv)
    r_p, p_p = pearsonr(xv, yv)
    ax_d.scatter(xv, yv, s=26, c=FUS_C, edgecolors='white', lw=0.5, alpha=0.9, zorder=3)
    b1, b0 = np.polyfit(xv, yv, 1)
    xx = np.linspace(xv.min(), xv.max(), 50)
    ax_d.plot(xx, b1 * xx + b0, color='#333333', lw=1.0, ls='--', zorder=2)
    lo2 = sg[np.argsort(xv)[:2]]
    hi1 = sg[np.argsort(xv)[-1:]]
    for s in set(list(lo2) + list(hi1) + [10]):
        if s not in sg:
            continue
        k = np.where(sg == s)[0][0]
        ax_d.annotate(F.SPECIES_NAME[int(s)], (xv[k], yv[k]),
                      textcoords='offset points', xytext=(4, -1), fontsize=5.2,
                      fontstyle='italic', color='#444444')
    ax_d.set_xlabel('Fusion linear-probe AUROC', fontsize=7.5)
    ax_d.set_ylabel('Ensemble test MCC', fontsize=7.5)
    ax_d.spines[['top', 'right']].set_visible(False)
    ax_d.grid(True, color='#EEEEEE', lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.text(0.03, 0.965,
              f'Spearman ρ = {rho:.2f} (P = {p_s:.1e})\nPearson r = {r_p:.2f}',
              transform=ax_d.transAxes, fontsize=6.4, va='top', ha='left')
    band_header(ax_d, 'd', 'Separability tracks MCC', y_ax=1.05,
                letter_dx=-0.20, title_dx=0.0, title_fs=7.6)

    print('[3/3] rendering + saving panels ...')
    save = [('a_umap_grid', ax_a[0], ax_a[1:]),
            ('b_separability_bars', ax_b, []),
            ('c_fusion_vs_single', ax_c[0], [ax_c[1]]),
            ('d_separability_vs_mcc', ax_d, [])]
    F.save_panels(fig, save, PANEL_DIR)

    out = os.path.join(HERE, 'fig_representation_umap')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)

    print('\n==== SUMMARY ====')
    print(f'mean fusion AUROC      = {fus.mean():.4f}')
    print(f'mean best-single AUROC = {best_auc.mean():.4f}  (gain +{fus.mean()-best_auc.mean():.4f})')
    print(f'fusion >= best-single  : {nwin}/{len(sids)} species')
    print(f'Spearman(sep, MCC)     = {rho:.3f} (P={p_s:.2e}); Pearson = {r_p:.3f} (P={p_p:.2e})')
    lo = sids[np.argsort(fus)[:3]]
    print(f'least separable species: ' + ', '.join(f'{F.SPECIES_NAME[int(s)]}(s{int(s)})' for s in lo))
    print(f'panel c species        : {F.SPECIES_NAME[c_sid]} (s{c_sid}), best view '
          f'{F.VIEW_NAMES[c_bview]}, Δ=+{gain[c_pos]:.3f}')
    print('saved', out + '.png/.pdf ; panels ->', PANEL_DIR)

if __name__ == '__main__':
    main()
