import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import logomaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _figlib import set_style, SPECIES_NAME, save_panels

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DROOT = os.path.join(ROOT, 'data', 'Benchmark Dataset', 'csv')
BASES = 'ACGT'
L = 81
TSS = 60
X0, X1 = 15, 66
EPS = 1e-9
BOX35 = (21, 27)
BOX10 = (48, 55)
NT_COLORS = {'A': '#2E8B57', 'C': '#1F6FB2', 'G': '#E69F00', 'T': '#C0392B'}

def load_species(sid):
    frames = []
    for p in [f'{DROOT}/Train_5fold/{sid}/fold_1_train.csv',
              f'{DROOT}/Train_5fold/{sid}/fold_1_val.csv',
              f'{DROOT}/Test/{sid}_test.csv']:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames).drop_duplicates('text')

def freq(seqs):
    M = np.zeros((4, L))
    for s in seqs:
        for i, ch in enumerate(s):
            j = BASES.find(ch)
            if j >= 0:
                M[j, i] += 1
    return M / max(len(seqs), 1)

def pos_neg_freq(df):
    return freq(df[df.label == 1]['text'].values), freq(df[df.label == 0]['text'].values)

def kl_profile(fp, fn):
    return np.sum(fp * np.log2((fp + EPS) / (fn + EPS)), axis=0)

def rel(idx):
    r = idx - TSS
    return r if r < 0 else r + 1

def main():
    set_style()
    rng = np.arange(X0, X1)
    rel_ticks = np.array([rel(i) for i in rng])

    eco = load_species(10)
    fp, fn = pos_neg_freq(eco)
    n_pos = int((eco.label == 1).sum())
    kl_eco = kl_profile(fp, fn)

    sids = list(range(1, 24))
    kl_mat = []
    for sid in sids:
        df = load_species(sid)
        a, b = pos_neg_freq(df)
        kl_mat.append(kl_profile(a, b))
    kl_mat = np.array(kl_mat)
    order = np.argsort(kl_mat[:, X0:X1].sum(1))[::-1]
    kl_mat = kl_mat[order]
    sp_labels = [f'{SPECIES_NAME[sids[i]]} ({sids[i]})' for i in order]

    fig = plt.figure(figsize=(7.1, 8.4))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.05, 0.85, 2.6],
                  hspace=0.26, left=0.135, right=0.905, top=0.945, bottom=0.075)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    diff = (fp - fn).T[X0:X1]
    dfm = pd.DataFrame(diff, columns=list(BASES))
    dfm.index = rng
    logo = logomaker.Logo(dfm, ax=ax_a, color_scheme=NT_COLORS,
                          flip_below=True, shade_below=0.0, fade_below=0.0,
                          font_name='Arial')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True)
    ax_a.axhline(0, color='#333333', lw=0.6)
    ax_a.set_ylabel('Enrichment\n(P$_{+}$ − P$_{-}$)', fontsize=7)
    ax_a.set_xlim(X0 - 0.5, X1 - 0.5)
    ax_a.set_xticks([])

    ax_b.fill_between(rng, kl_eco[X0:X1], color='#5B6C7D', alpha=0.35, lw=0, zorder=2)
    ax_b.plot(rng, kl_eco[X0:X1], color='#34495E', lw=1.1, zorder=3)
    ax_b.set_ylabel('Information\n(bits, KL)', fontsize=7)
    ax_b.set_xlim(X0 - 0.5, X1 - 0.5)
    ax_b.set_ylim(0, None)
    ax_b.set_xticks([])
    ax_b.spines[['top', 'right']].set_visible(False)

    im = ax_c.imshow(kl_mat[:, X0:X1], aspect='auto', cmap='magma',
                     vmin=0, vmax=np.percentile(kl_mat[:, X0:X1], 99.0),
                     extent=[X0 - 0.5, X1 - 0.5, len(sids) - 0.5, -0.5],
                     interpolation='nearest')
    ax_c.set_yticks(np.arange(len(sids)))
    ax_c.set_yticklabels(sp_labels, fontsize=5.6, fontstyle='italic')
    ax_c.set_ylabel('Species (sorted by motif strength)', fontsize=7)
    show_rel = [-45, -35, -25, -15, -10, -5, 1, 5]
    show_idx = [TSS + (r if r < 0 else r - 1) for r in show_rel]
    show_idx = [i for i in show_idx if X0 <= i < X1]
    ax_c.set_xticks(show_idx)
    ax_c.set_xticklabels([f'{rel(i):+d}' for i in show_idx], fontsize=6.3)
    ax_c.set_xlabel('Position relative to transcription start site (+1)', fontsize=7.5)
    cax = fig.add_axes([0.912, 0.075, 0.016, 0.30])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('KL (bits)', fontsize=6.3)
    cb.ax.tick_params(labelsize=5.6)

    arch = Rectangle((28.5, -0.5), 8, 2, fill=False, edgecolor='#2EC4C6',
                     lw=0.9, ls=(0, (3, 2)), zorder=6)
    ax_c.add_patch(arch)
    ax_c.annotate('Archaeal TATA-box', xy=(32.5, 1.5), xytext=(18, 4.3),
                  fontsize=6.0, color='#1f9a9c', fontweight='bold', ha='left',
                  arrowprops=dict(arrowstyle='-|>', color='#1f9a9c', lw=0.8,
                                  shrinkA=1, shrinkB=2))

    for ax in (ax_a, ax_b, ax_c):
        for (lo, hi), c in [(BOX35, '#888888'), (BOX10, '#C1272D')]:
            ax.axvspan(lo - 0.5, hi - 0.5, color=c, alpha=0.10, lw=0, zorder=0)
        ax.axvline(TSS, color='#2E8B57', lw=0.8, ls=(0, (3, 2)), zorder=1, alpha=0.8)

    ax_a.text((BOX10[0] + BOX10[1]) / 2 - 0.5, ax_a.get_ylim()[1] * 0.98, '–10 box',
              ha='center', va='top', fontsize=6.5, color='#C1272D', fontweight='bold')
    ax_a.text((BOX35[0] + BOX35[1]) / 2 - 0.5, ax_a.get_ylim()[1] * 0.98, '–35 box',
              ha='center', va='top', fontsize=6.5, color='#555555', fontweight='bold')
    ax_a.text(TSS, ax_a.get_ylim()[1] * 0.98, '+1', ha='center', va='top',
              fontsize=6.5, color='#2E8B57', fontweight='bold')

    def ptitle(ax, letter, text, y=1.02):
        ax.set_title(letter, loc='left', fontsize=11, fontweight='bold', x=-0.115, y=y - 0.02)
        ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.6, fontweight='bold', va='bottom')
    ptitle(ax_a, 'a', f'Two-sample sequence logo — E. coli ($n_{{+}}$ = {n_pos})')
    ptitle(ax_b, 'b', 'Positional discriminative information (E. coli)')
    ptitle(ax_c, 'c', 'Conservation of promoter elements across 23 prokaryotes')

    save_panels(fig, [
        ('a_two_sample_logo',       ax_a, []),
        ('b_positional_information', ax_b, []),
        ('c_conservation_heatmap',  ax_c, [cax]),
    ], os.path.join(os.path.dirname(__file__), 'panel'))

    out = os.path.join(os.path.dirname(__file__), 'fig_sequence_motif')
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', out + '.png')

if __name__ == '__main__':
    main()
