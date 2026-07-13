import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, matthews_corrcoef)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _figlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = np.linspace(0, 1, 200)
HERO = F.HERO
BEST_BL = 'msBERT'
OVERLAY = ['HyenaDNA', 'GROVER', 'DNABERT2-CAMP', 'msBERT']
METHODS = [HERO] + F.BASELINES
GREY = '#8a8a8a'
FAN_COLOR = '#E2A6A9'

SPLITS = {
    'test': ('fig_roc_pr',     'panel',     'Test set'),
    'val':  ('fig_roc_pr_val', 'panel_val', 'Validation (out-of-fold)'),
}

def collect(method, split):
    out = []
    for sid in F.ALL_SPECIES:
        if split == 'test':
            yt, pr = F.load_species_ensemble(method, sid, 'test')
        else:
            yt, pr = F.load_species_oof(method, sid)
        if yt is not None:
            out.append((yt, pr))
    return out

def macro_roc(data):
    tprs, aucs = [], []
    for yt, pr in data:
        fpr, tpr, _ = roc_curve(yt, pr)
        t = np.interp(GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(auc(fpr, tpr))
    tprs = np.array(tprs)
    return tprs.mean(0), tprs.std(0), np.array(aucs)

def macro_pr(data):
    precs, aps = [], []
    for yt, pr in data:
        p, r, _ = precision_recall_curve(yt, pr)
        order = np.argsort(r)
        precs.append(np.interp(GRID, r[order], p[order]))
        aps.append(average_precision_score(yt, pr))
    precs = np.array(precs)
    return precs.mean(0), precs.std(0), np.array(aps)

def reliability(data, n_bins=12):
    yt = np.concatenate([d[0] for d in data])
    pr = np.concatenate([d[1] for d in data])
    return _reliability_arr(yt, pr, n_bins)

def _reliability_arr(yt, pr, n_bins):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pr, bins) - 1, 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(pr[m].mean())
        ys.append(yt[m].mean())
        ws.append(m.sum())
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    ece = np.sum(ws / ws.sum() * np.abs(xs - ys)) if ws.sum() else np.nan
    return xs, ys, ece

def per_species_ece(data, n_bins=10):
    return np.array([_reliability_arr(yt, pr, n_bins)[2] for yt, pr in data])

def macro_mcc_sweep(data, thr):
    means, sds = [], []
    for t in thr:
        vals = []
        for yt, pr in data:
            pred = (pr >= t).astype(int)
            vals.append(matthews_corrcoef(yt, pred) if len(np.unique(yt)) > 1 else np.nan)
        vals = np.array(vals)
        means.append(np.nanmean(vals))
        sds.append(np.nanstd(vals))
    return np.array(means), np.array(sds)

def main():
    F.set_style()
    for split in SPLITS:
        build(split)

def build(split):
    out_name, panel_dir, tag = SPLITS[split]
    print(f'[{split}] loading predictions ...')
    DATA = {m: collect(m, split) for m in METHODS}
    for m in METHODS:
        n = sum(len(d[0]) for d in DATA[m])
        print(f'  {m}: {len(DATA[m])} species, {n} samples')

    macro_r = {m: macro_roc(DATA[m]) for m in METHODS}
    macro_p = {m: macro_pr(DATA[m]) for m in METHODS}

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.28,
                  left=0.085, right=0.975, top=0.925, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    axin_a = panel_roc(ax_a, DATA, macro_r)
    axin_b = panel_pr(ax_b, DATA, macro_p)
    panel_calibration(ax_c, DATA, split)
    panel_threshold(ax_d, DATA)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_aspect('equal', adjustable='box')

    fig.text(0.978, 0.968, tag, ha='right', va='top', fontsize=7.5,
             color='#555555', fontstyle='italic')

    F.save_panels(fig, [
        ('a_roc',         ax_a, [axin_a]),
        ('b_pr',          ax_b, [axin_b]),
        ('c_calibration', ax_c, []),
        ('d_threshold',   ax_d, []),
    ], os.path.join(HERE, panel_dir))

    out = os.path.join(HERE, out_name)
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved', out + '.png / .pdf')

def panel_roc(ax, DATA, macro_r):
    dp = DATA[HERO]
    for yt, pr in dp:
        fpr, tpr, _ = roc_curve(yt, pr)
        ax.plot(fpr, tpr, color=FAN_COLOR, lw=0.5, alpha=0.5, zorder=2)
    mtb, _, _ = macro_r[BEST_BL]
    ax.plot(GRID, mtb, color=F.COLORS[BEST_BL], lw=1.3, zorder=4)
    mt, st, aucs = macro_r[HERO]
    ax.fill_between(GRID, mt - st, np.minimum(mt + st, 1), color=F.COLORS[HERO],
                    alpha=0.15, lw=0, zorder=3)
    ax.plot(GRID, mt, color=F.COLORS[HERO], lw=1.9, zorder=6)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')

    axin = ax.inset_axes([0.40, 0.14, 0.565, 0.56])
    for m in OVERLAY + [HERO]:
        mm, _, aa = macro_r[m]
        lw = 1.9 if m == HERO else 1.2
        axin.plot(GRID, mm, color=F.COLORS[m], lw=lw, zorder=(6 if m == HERO else 4),
                  label=f'{F.DISPLAY[m]} ({aa.mean():.3f})')
    axin.set_xlim(0, 0.40); axin.set_ylim(0.60, 1.005)
    axin.set_xticks([0, 0.2, 0.4]); axin.set_yticks([0.6, 0.8, 1.0])
    _style_inset(axin)
    ax.indicate_inset_zoom(axin, edgecolor='#9a9a9a', lw=0.6, alpha=0.8)
    leg = axin.legend(loc='lower right', fontsize=5.1, title='Macro AUROC',
                      title_fontsize=5.3, handlelength=1.0, borderpad=0.28,
                      labelspacing=0.20, handletextpad=0.45, frameon=True,
                      framealpha=0.93, edgecolor='#D9D9D9')
    leg.get_frame().set_linewidth(0.4)
    panel_title(ax, 'a', 'ROC across 23 species')
    print(f'  [a] macro-AUROC: ' +
          ', '.join(f'{F.DISPLAY[m]}={macro_r[m][2].mean():.3f}' for m in METHODS))
    return axin

def panel_pr(ax, DATA, macro_p):
    dp = DATA[HERO]
    for yt, pr in dp:
        p, r, _ = precision_recall_curve(yt, pr)
        ax.plot(r, p, color=FAN_COLOR, lw=0.5, alpha=0.5, zorder=2)
    mpb, _, _ = macro_p[BEST_BL]
    ax.plot(GRID, mpb, color=F.COLORS[BEST_BL], lw=1.3, zorder=4)
    mp, spr, aps = macro_p[HERO]
    ax.fill_between(GRID, mp - spr, np.minimum(mp + spr, 1), color=F.COLORS[HERO],
                    alpha=0.15, lw=0, zorder=3)
    ax.plot(GRID, mp, color=F.COLORS[HERO], lw=1.9, zorder=6)
    base = np.mean([yt.mean() for yt, _ in dp])
    ax.axhline(base, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(0.975, base - 0.025, f'No-skill ({base:.2f})', fontsize=5.6, color='#777777',
            va='top', ha='right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')

    axin = ax.inset_axes([0.115, 0.15, 0.565, 0.56])
    for m in OVERLAY + [HERO]:
        mm, _, aa = macro_p[m]
        lw = 1.9 if m == HERO else 1.2
        axin.plot(GRID, mm, color=F.COLORS[m], lw=lw, zorder=(6 if m == HERO else 4),
                  label=f'{F.DISPLAY[m]} ({aa.mean():.3f})')
    axin.set_xlim(0.55, 1.0); axin.set_ylim(0.55, 1.005)
    axin.set_xticks([0.6, 0.8, 1.0]); axin.set_yticks([0.6, 0.8, 1.0])
    _style_inset(axin)
    ax.indicate_inset_zoom(axin, edgecolor='#9a9a9a', lw=0.6, alpha=0.8)
    leg = axin.legend(loc='lower left', fontsize=5.1, title='Macro AUPRC',
                      title_fontsize=5.3, handlelength=1.0, borderpad=0.28,
                      labelspacing=0.20, handletextpad=0.45, frameon=True,
                      framealpha=0.93, edgecolor='#D9D9D9')
    leg.get_frame().set_linewidth(0.4)
    panel_title(ax, 'b', 'Precision-recall across 23 species')
    print(f'  [b] macro-AUPRC: ' +
          ', '.join(f'{F.DISPLAY[m]}={macro_p[m][2].mean():.3f}' for m in METHODS))
    return axin

def panel_calibration(ax, DATA, split):
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    pe = per_species_ece(DATA[HERO])
    wi = int(np.argmax(pe))
    wsid = F.ALL_SPECIES[wi]
    wyt, wpr = DATA[HERO][wi]
    wx, wy, _ = _reliability_arr(wyt, wpr, 8)
    ax.plot(wx, wy, color=F.COLORS[HERO], lw=0.9, ls=(0, (2, 1.5)), alpha=0.7,
            marker='.', ms=3, zorder=3)
    eces = {}
    for m in [BEST_BL, HERO]:
        xs, ys, ece = reliability(DATA[m])
        eces[m] = ece
        ax.plot(xs, ys, color=F.COLORS[m], lw=1.5, marker=F.MARKERS[m], ms=3.4,
                mfc='white', mew=0.8, zorder=(6 if m == HERO else 5),
                label=f'{F.DISPLAY[m]} (ECE {ece:.3f})')
    ax.annotate(f'Worst-calibrated species:\n{F.SPECIES_NAME[wsid]} (ECE {pe[wi]:.3f})',
                xy=(wx[np.argmax(np.abs(wx - wy))], wy[np.argmax(np.abs(wx - wy))]),
                xytext=(0.50, 0.10), fontsize=5.4, color=F.COLORS[HERO], va='bottom',
                ha='left', style='italic',
                arrowprops=dict(arrowstyle='-', color=F.COLORS[HERO], lw=0.5, alpha=0.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency')
    leg = ax.legend(loc='upper left', fontsize=6.0, handlelength=1.4)
    leg.set_zorder(10)
    panel_title(ax, 'c', 'Probability calibration (raw, pooled)')
    print(f'  [c] RAW ECE: ' + ', '.join(f'{F.DISPLAY[m]}={eces[m]:.3f}' for m in [HERO, BEST_BL]) +
          f'  | worst {F.SPECIES_NAME[wsid]}={pe[wi]:.3f}')

def panel_threshold(ax, DATA):
    thr = np.linspace(0.02, 0.98, 97)
    sweeps = {m: macro_mcc_sweep(DATA[m], thr) for m in METHODS}
    for m in ['HyenaDNA', 'GROVER', 'DNABERT2-CAMP']:
        mm, _ = sweeps[m]
        ax.plot(thr, mm, color=F.COLORS[m], lw=0.9, alpha=0.75, zorder=3)
    mmb, _ = sweeps[BEST_BL]
    ax.plot(thr, mmb, color=F.COLORS[BEST_BL], lw=1.5, zorder=5,
            label=f'{F.DISPLAY[BEST_BL]}')
    mm, ss = sweeps[HERO]
    ax.plot(thr, mm, color=F.COLORS[HERO], lw=1.9, zorder=6, label=f'{F.DISPLAY[HERO]}')
    ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(0.485, 0.30, 'Default 0.5', fontsize=5.4, color='#777777', rotation=90,
            va='bottom', ha='right')
    for m in [BEST_BL, HERO]:
        mmm, _ = sweeps[m]
        k = int(np.nanargmax(mmm))
        ax.plot(thr[k], mmm[k], 'o', color=F.COLORS[m], ms=4, mfc='white', mew=1.0, zorder=7)

    lead = np.all(sweeps[HERO][0] >= sweeps[BEST_BL][0] - 1e-9)
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.86, np.nanmax(mm) + 0.06))
    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Macro MCC')
    ax.legend(loc='lower center', fontsize=6.0, handlelength=1.4, ncol=1,
              borderpad=0.3, labelspacing=0.25)
    panel_title(ax, 'd', 'MCC across decision thresholds')
    print(f'  [d] hero peak macroMCC={np.nanmax(sweeps[HERO][0]):.3f} @thr'
          f'={thr[int(np.nanargmax(sweeps[HERO][0]))]:.2f}; '
          f'{BEST_BL} peak={np.nanmax(sweeps[BEST_BL][0]):.3f}; '
          f'hero>=best at all thresholds: {bool(lead)}')

def _style_inset(axin):
    axin.tick_params(labelsize=5.2, length=2)
    for s in axin.spines.values():
        s.set_edgecolor('#9a9a9a'); s.set_linewidth(0.6)
    axin.set_facecolor('#FCFCFC')

def panel_title(ax, letter, text):
    ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=-0.13, y=1.0)
    ax.text(0.0, 1.02, text, transform=ax.transAxes, fontsize=7.5,
            fontweight='bold', va='bottom')

if __name__ == '__main__':
    main()
