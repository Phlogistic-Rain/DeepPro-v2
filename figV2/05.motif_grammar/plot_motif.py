from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from _figlib import set_style

BASES = list("ACGT")
BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}

def ic_logo_df(prob_pos4: np.ndarray) -> pd.DataFrame:
    eps = 1e-9
    ic = 2.0 + (prob_pos4 * np.log2(prob_pos4 + eps)).sum(axis=1)
    return pd.DataFrame(prob_pos4 * ic[:, None], columns=BASES)

def consensus_ref_df(cons: str) -> pd.DataFrame:
    h = np.zeros((len(cons), 4), dtype=np.float32)
    for i, ch in enumerate(cons):
        h[i, BASES.index(ch)] = 2.0
    return pd.DataFrame(h, columns=BASES)

def draw_logo(ax, df, title, color="#1a1a1a"):
    logomaker.Logo(df, ax=ax, color_scheme=BASE_COLORS, show_spines=False)
    ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([0, 1, 2])
    ax.set_title(title, fontsize=6.3, pad=2, color=color)
    ax.tick_params(labelsize=6)

def main():
    set_style()
    D = np.load(os.path.join(HERE, "motif_data.npz"), allow_pickle=True)
    S = np.load(os.path.join(HERE, "shuffle_data.npz"), allow_pickle=True)
    P10, P35 = D["P10"], D["P35"]
    r10, r35 = D["r10"], D["r35"]
    c10, c35 = [str(x) for x in D["c10"]], [str(x) for x in D["c35"]]
    cons10, cons35 = str(D["cons10"]), str(D["cons35"])
    K = P10.shape[0]
    order10 = np.argsort(-r10)
    top10 = [e for e in order10 if r10[e] > 0][:4]
    best35 = int(D["best35"])

    P10s = S["P10_shuffle"]; best10s = int(S["best10_shuffle"])
    null_best_r = S["null_best_r"]; dd_win = S["dd_win"]
    dd_cons = str(S["dd_consensus"]); r_dd = float(S["r_dd"])
    real_best_r = float(r10.max())

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 1.05],
                          hspace=1.05, wspace=0.5,
                          left=0.065, right=0.985, top=0.86, bottom=0.085)

    fig.text(0.006, 0.93, "a", fontsize=11, fontweight="bold")
    fig.text(0.52, 0.925, "−10 box (TATAAT) recovered by grammar experts  ·  standalone capability proof",
             fontsize=8.0, ha="center", fontweight="bold")
    axRef = fig.add_subplot(gs[0, 0])
    draw_logo(axRef, consensus_ref_df(cons10), f"σ70 −10 consensus\n{cons10}", color="#555555")
    axRef.set_ylabel("bits", fontsize=6.5)
    for j, e in enumerate(top10):
        ax = fig.add_subplot(gs[0, j + 1])
        draw_logo(ax, ic_logo_df(P10[e].T), f"Expert {e}  ({c10[e]})\nr = {r10[e]:+.2f}")

    axB = fig.add_subplot(gs[1, 0:3])
    xs = np.arange(K)
    cols = ["#C1272D" if r10[e] > 0 else "#9e9e9e" for e in range(K)]
    axB.bar(xs, r10, color=cols, width=0.66, edgecolor="#333333", linewidth=0.4)
    axB.axhline(0, color="#333333", lw=0.6)
    axB.set_xticks(xs); axB.set_xticklabels([f"E{e}" for e in range(K)], fontsize=6.5)
    axB.set_ylabel("PWM corr. with\n−10 consensus (r)", fontsize=6.5)
    axB.set_ylim(-0.5, 0.8)
    for e in range(K):
        y = r10[e] + (0.03 if r10[e] >= 0 else -0.06)
        axB.text(e, y, f"{r10[e]:+.2f}", ha="center", fontsize=5.5, color="#333333")
    axB.set_title(f"−10 box: {int((r10 > 0).sum())}/{K} experts positively correlated "
                  f"(best E{int(np.argmax(r10))}, r={r10.max():+.2f})", fontsize=7)
    fig.text(0.006, 0.60, "b", fontsize=11, fontweight="bold")

    axC = fig.add_subplot(gs[1, 3:5])
    draw_logo(axC, ic_logo_df(P35[best35].T),
              f"Best −35: Expert {best35} ({c35[best35]})\nvs {cons35}, r = {r35[best35]:+.2f} (weaker)")
    axC.set_ylabel("bits", fontsize=6.5)
    fig.text(0.60, 0.60, "c", fontsize=11, fontweight="bold")

    fig.text(0.006, 0.30, "d", fontsize=11, fontweight="bold")
    fig.text(0.52, 0.295, "Label-shuffle control: the −10 recovery is label-driven, not a sharpness artifact",
             fontsize=7.6, ha="center", fontweight="bold")
    axD = fig.add_subplot(gs[2, 0:3])
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(null_best_r)) - 0.5) * 0.18
    axD.scatter(null_best_r, np.zeros_like(null_best_r) + jitter, s=26, color="#9e9e9e",
                edgecolor="#555555", linewidth=0.4, zorder=3, label="Shuffled labels (5 seeds)")
    axD.axvline(float(null_best_r.mean()), color="#9e9e9e", ls="--", lw=0.8, zorder=2)
    axD.scatter([real_best_r], [0], s=70, marker="*", color="#C1272D",
                edgecolor="white", linewidth=0.5, zorder=5, label="Real labels (best expert)")
    axD.annotate(f"real\nr={real_best_r:+.2f}", xy=(real_best_r, 0), xytext=(real_best_r, 0.28),
                 ha="center", fontsize=5.8, color="#C1272D", fontweight="bold")
    axD.annotate(f"shuffle null\nmean {null_best_r.mean():+.2f}", xy=(float(null_best_r.mean()), 0),
                 xytext=(float(null_best_r.mean()), -0.34), ha="center", fontsize=5.6, color="#555555")
    axD.set_xlim(-0.05, 0.8); axD.set_ylim(-0.5, 0.5)
    axD.set_yticks([]); axD.set_xlabel("best-of-6 expert PWM corr. with −10 consensus", fontsize=6.2)
    axD.set_title("Real recovery exceeds shuffle null", fontsize=6.8)
    axD.spines[["top", "right", "left"]].set_visible(False)
    axD.legend(fontsize=5.2, loc="upper left", handletextpad=0.3, borderpad=0.3)

    axD2 = fig.add_subplot(gs[2, 3:5])
    draw_logo(axD2, ic_logo_df(P10s[best10s].T),
              f"Shuffle-label best −10  (r={float(null_best_r[0]):+.2f})\n"
              f"sharp but NOT TATAAT", color="#777777")
    axD2.set_ylabel("bits", fontsize=6.2)

    out = os.path.join(HERE, "fig_motif_grammar")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"[saved] {out}.png / .pdf")
    print(f"real best r={real_best_r:.2f}  shuffle null mean={null_best_r.mean():.2f} "
          f"max={null_best_r.max():.2f}  data-derived {dd_cons} r={r_dd:.2f}")

if __name__ == "__main__":
    main()
