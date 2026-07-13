import os
import re
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _figlib import set_style, save_panels

PANEL_DIR = 'panel'
XLSX = '../../paper/Result.xlsx'

METRICS = ['ACC', 'Sn', 'Sp', 'Pre', 'F1', 'MCC', 'AUC']
METRICS7 = ['ACC', 'Sn', 'Sp', 'Pre', 'F1', 'MCC', 'AUC']

HERO = 'DeepPro-v2'
ORDER = ['DeepPro-v2', 'msBERT', 'DNABERT2-CAMP', 'GROVER', 'HyenaDNA',
         'iPro-MP', 'Prompt', 'PromoterLCNN', 'iPro-WAEL']
FM_METHODS = ['DeepPro-v2', 'msBERT', 'DNABERT2-CAMP', 'GROVER', 'HyenaDNA']
EXCLUDE = {'ProMLK', 'deepPro'}

COLORS = {
    'DeepPro-v2':    '#C1272D',
    'msBERT':        '#0072B2',
    'DNABERT2-CAMP': '#56B4E9',
    'GROVER':        '#009E73',
    'HyenaDNA':      '#E69F00',
    'iPro-MP':       '#CC79A7',
    'Prompt':        '#9467BD',
    'PromoterLCNN':  '#8C7B6B',
    'iPro-WAEL':     '#7F7F7F',
}

def load_sheet(ws):
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    blocks = []
    for c, v in enumerate(hdr):
        if v and isinstance(v, str) and '(' in v:
            m = re.search(r'\(([^)]+)\)', v)
            blocks.append((c, m.group(1).strip()))
    method_order = [k for _, k in blocks]
    species = []
    for r in rows[2:]:
        if not isinstance(r[0], (int, float)):
            continue
        rec = {'id': int(r[0]), 'name': r[1], 'data': {}}
        for start, key in blocks:
            rec['data'][key] = {METRICS[i]: r[start + i] for i in range(7)}
        species.append(rec)
    return species, method_order

def mean_metric(species, method, metric):
    vals = [s['data'][method][metric] for s in species
            if s['data'][method].get(metric) is not None]
    return float(np.mean(vals)) if vals else np.nan

def std_metric(species, method, metric):
    vals = [s['data'][method][metric] for s in species
            if s['data'][method].get(metric) is not None]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

def abbrev(name, sid):
    parts = name.strip().split()
    return f"{parts[0][0]}. {parts[1]} ({sid})" if len(parts) >= 2 else f"{name} ({sid})"

ERRKW = dict(ecolor='#4D4D4D', elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=5)

def draw_panel_a(ax, species_sorted, methods, split, panel_label='a', title_x=-0.30):
    M = np.array([[s['data'][m]['MCC'] for m in methods] for s in species_sorted], float)
    n_sp, n_m = M.shape
    im = ax.imshow(M, aspect='auto', cmap='YlGnBu', vmin=0.3, vmax=1.0)
    for i in range(n_sp):
        row = M[i]
        jmax = int(np.nanargmax(row))
        for j in range(n_m):
            v = row[j]
            txt = f'{v:.2f}'.lstrip('0') if v < 1 else '1.0'
            ax.text(j, i, txt, ha='center', va='center', fontsize=4.6,
                    color='white' if v > 0.72 else '#1a1a1a', zorder=3)
        ax.text(jmax, i - 0.32, '*', ha='center', va='center', fontsize=7,
                color='#C1272D' if methods[jmax] == HERO else '#333333', zorder=4)
    jh = methods.index(HERO)
    ax.add_patch(Rectangle((jh - 0.5, -0.5), 1, n_sp, fill=False,
                           edgecolor='#C1272D', lw=1.6, zorder=5, clip_on=False))
    ax.set_xticks(range(n_m))
    xt = ax.set_xticklabels([m for m in methods], rotation=40, ha='right', fontsize=6.2)
    xt[jh].set_color('#C1272D'); xt[jh].set_fontweight('bold')
    ax.set_yticks(range(n_sp))
    ax.set_yticklabels([abbrev(s['name'], s['id']) for s in species_sorted],
                       fontsize=5.8, fontstyle='italic')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label('MCC', fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    if panel_label:
        ax.set_title(panel_label, loc='left', fontsize=11, fontweight='bold', x=title_x, y=1.003)
    ax.text(0.0, 1.008, f'Per-species MCC across 23 prokaryotes ({split})',
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    return cb

def draw_panel_b(ax, species, panel_label='b', title_x=-0.20):
    nb = len(FM_METHODS)
    bw = 0.82 / nb
    xidx = np.arange(len(METRICS7))
    for j, m in enumerate(FM_METHODS):
        vals = [mean_metric(species, m, k) for k in METRICS7]
        errs = [std_metric(species, m, k) for k in METRICS7]
        ax.bar(xidx + (j - (nb - 1) / 2) * bw, vals, bw, color=COLORS[m],
               edgecolor='white', linewidth=0.3, label=m, zorder=3,
               yerr=errs, error_kw=ERRKW)
    ax.set_xticks(xidx)
    ax.set_xticklabels(METRICS7, fontsize=6.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_ylabel('Score', fontsize=7.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#EAEAEA', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=5,
              frameon=False, fontsize=5.8, columnspacing=1.0,
              handlelength=1.0, handletextpad=0.4)
    if panel_label:
        ax.set_title(panel_label, loc='left', fontsize=11, fontweight='bold', x=title_x, y=1.005)
    ax.text(0.0, 1.02, 'Seven metrics, foundation-model methods',
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')

def draw_panel_c(ax, species, methods, panel_label='c', title_x=-0.42):
    mm = sorted([(m, mean_metric(species, m, 'MCC')) for m in methods], key=lambda t: t[1])
    cy = np.arange(len(mm))
    for yi, (m, v) in zip(cy, mm):
        err = std_metric(species, m, 'MCC')
        ax.barh(yi, v, color=COLORS[m], edgecolor='white', linewidth=0.4,
                height=0.66, zorder=3, xerr=err, error_kw=ERRKW)
        ax.text(0.015, yi, f'{v:.3f}', va='center', ha='left', fontsize=6.0,
                color='white', fontweight='bold', zorder=6)
    ax.set_yticks(cy)
    yl = ax.set_yticklabels([m for m, _ in mm], fontsize=6.8)
    for t, (m, _) in zip(yl, mm):
        if m == HERO:
            t.set_color('#C1272D'); t.set_fontweight('bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Average MCC', fontsize=7.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color='#EAEAEA', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if panel_label:
        ax.set_title(panel_label, loc='left', fontsize=11, fontweight='bold', x=title_x, y=1.005)
    ax.text(0.0, 1.03, 'Average MCC ranking', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom')

def make_figure(split, species, order_ids, panel_prefix, panel_dir=PANEL_DIR):
    methods = [m for m in ORDER if m not in EXCLUDE]
    id2rec = {s['id']: s for s in species}
    species_sorted = [id2rec[i] for i in order_ids]

    fig = plt.figure(figsize=(7.4, 8.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.05], width_ratios=[1.28, 1.0],
                  hspace=0.26, wspace=0.42, left=0.16, right=0.965, top=0.945, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    cb = draw_panel_a(ax_a, species_sorted, methods, split)
    draw_panel_b(ax_b, species)
    draw_panel_c(ax_c, species, methods)

    if panel_prefix:
        save_panels(fig, [
            (f'{panel_prefix}_a_mcc_heatmap', ax_a, [cb.ax]),
            (f'{panel_prefix}_b_seven_metrics', ax_b, []),
            (f'{panel_prefix}_c_mcc_ranking', ax_c, []),
        ], panel_dir)
    return fig

def main():
    set_style()
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    sp_test, _ = load_sheet(wb['test'])
    order_ids = [s['id'] for s in sorted(sp_test, key=lambda r: r['data'][HERO]['MCC'] or 0)]

    fig = make_figure('Independent test', sp_test, order_ids, 'test')
    fig.savefig('fig_performance.png', dpi=600, bbox_inches='tight')
    fig.savefig('fig_performance.pdf', bbox_inches='tight')
    plt.close(fig)

    sp_val, _ = load_sheet(wb['val'])
    figv = make_figure('Cross-validation', sp_val, order_ids, 'val', panel_dir='panel_val')
    figv.savefig('fig_performance_val.png', dpi=600, bbox_inches='tight')
    figv.savefig('fig_performance_val.pdf', bbox_inches='tight')
    plt.close(figv)
    print('done: fig_performance[.pdf/.png] (+_val), panels in', PANEL_DIR)

if __name__ == '__main__':
    main()
