# DeepPro-v2: A Mixture-of-Experts Framework for Interpretable Prokaryotic Promoter Prediction

<p align="center">
  <a href="https://phlogistic-rain.github.io/DeepPro-v2/">
    <img src="assets/head.png" alt="DeepPro-v2 — interactive demo" width="820">
  </a>
</p>

<p align="center">
  <a href="https://phlogistic-rain.github.io/DeepPro-v2/"><b>▶ Explore the interactive demo →</b></a>
</p>

DeepPro‑v2 predicts prokaryotic promoters from 81‑bp DNA sequences by
aggregating **five** genomic language‑model backbones through a
soft mixture‑of‑experts fusion, with an interpretable σ‑factor *grammar* branch.

- 📄 **Paper**: `<PAPER_URL>` (TODO)
- 💾 **Data & weights (Zenodo)**: [zenodo.org/records/21344602](https://zenodo.org/records/21344602?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM2NWQ3YmMyLTdkMmEtNDFjMS1iNDlmLTYyOTQwY2FlZmYyYiIsImRhdGEiOnt9LCJyYW5kb20iOiI4YTVhNGI3OGZiMTE4NWQ4MWM2MDU4MTM3ZDk1Y2NjZiJ9.zgznyGzDRiuKhrH1wMfE68DVdpgFWpH-9pl2LoxQh7LtMRUXdJ715efByvgmlGTwHWs9MHK5AjEj-xKr5_9WvA)
- 🌐 **Interactive demo**: https://phlogistic-rain.github.io/DeepPro-v2/ 

---

## Results & analysis

DeepPro‑v2 was evaluated on a shared 23‑prokaryote benchmark (5‑fold, independent
test) against eight published baselines. The five headline figures are shown in
full below; six further analyses are in the collapsible section. Every figure is
also explorable at full resolution in the
[**interactive demo**](https://phlogistic-rain.github.io/DeepPro-v2/).

<p align="center">
  <img src="docs/assets/figV2/01.performance/fig_performance.png" width="860" alt="Per-species MCC across 23 prokaryotes; DeepPro-v2 ranks first with mean MCC 0.809">
</p>

**Fig. 1 · Benchmark across 23 prokaryotes.** DeepPro‑v2 attains the best mean
test MCC (**0.809**) of the nine methods evaluated — ahead of the strongest
external baseline, msBERT (0.767) — and leads the per‑species MCC map (**a**). It
wins 6 of 7 metrics among the foundation‑model methods (**b**) and holds the top
average‑MCC rank overall (**c**).

<p align="center">
  <img src="docs/assets/figV2/04.expert_complementarity/fig_expert_complementarity.png" width="860" alt="Five experts are complementary; their fusion beats every single expert">
</p>

**Fig. 2 · Why five experts (complementarity).** The five genomic language‑model
experts are genuinely complementary — per‑species probe performance shifts from
expert to expert (**a**), they disagree on distinct examples, and each contributes
uniquely‑correct predictions (**d**) — so their soft mixture‑of‑experts fusion
(linear‑probe AUROC 0.948) exceeds every single expert (best 0.921; **b**) and
narrows the gap toward an oracle upper bound (**c**).

<p align="center">
  <img src="docs/assets/figV2/05.motif_grammar/fig_motif_grammar.png" width="860" alt="Grammar experts recover the sigma-70 minus-10 box TATAAT; survives label-shuffle control">
</p>

**Fig. 3 · σ‑factor grammar recovered (standalone capability).** A standalone probe
of the grammar experts recovers the σ⁷⁰ −10 box (TATAAT): 5 of 6 experts correlate
positively with the −10 consensus (best *r* = +0.67; **a**, **b**), the recovery
survives a label‑shuffle control (real *r* = +0.67 vs shuffle‑null mean +0.29;
**d**), and a weaker −35 signal is also visible (**c**). This is a capability
demonstration of the grammar branch, not a claim about what the deployed
classifier attends to.

<p align="center">
  <img src="docs/assets/figV2/06.representation_umap/fig_representation_umap.png" width="860" alt="UMAP of fused representation separates promoters; separability tracks MCC">
</p>

**Fig. 4 · Learned representations.** UMAP of the fused representation separates
promoters from non‑promoters across species (**a**); the fusion exceeds the best
single‑expert view in nearly all species (mean AUROC 0.943, Δ up to +0.06;
**b**, **c**), and per‑species separability tracks ensemble test MCC very closely
(Spearman ρ = 0.89, Pearson *r* = 0.97; **d**).

<p align="center">
  <img src="docs/assets/figV2/09.cross_species/fig_cross_species_heatmaps.png" width="900" alt="23x23 cross-species transfer MCC organised by phylogeny; non-zero-shot">
</p>

**Fig. 5 · Cross‑species transfer.** A 23×23 transfer map (train on one species,
test on another) shows the self‑test diagonal is strongest and off‑diagonal
transfer is organised by phylogeny, under both genome‑tree and promoter‑tree
orderings. This is *non‑zero‑shot*: all species share the same TAPT‑adapted
backbones.

<details>
<summary><b>▸ Extended analyses — six more figures (calibration · significance · attribution · motifs · diversity · errors)</b></summary>

<br>

<table>
  <tr>
    <td width="50%"><img src="docs/assets/figV2/02.roc_pr_calibration/fig_roc_pr.png" width="100%" alt="ROC, precision-recall, calibration and threshold-MCC curves"></td>
    <td width="50%"><img src="docs/assets/figV2/03.statistical_significance/fig_significance.png" width="100%" alt="Friedman-Nemenyi diagram, pairwise Wilcoxon, per-species delta MCC"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Fig. S1 · Discrimination and calibration.</b> ROC and precision–recall across 23 species (macro AUROC 0.954, AUPRC 0.932), probability calibration (ECE 0.025), and MCC stability across decision thresholds.</sub></td>
    <td align="center"><sub><b>Fig. S2 · Statistical significance.</b> Friedman–Nemenyi critical‑difference diagram (P = 3.6×10⁻²², average rank 1.30, first of nine), pairwise Wilcoxon signed‑rank matrix, and per‑species ΔMCC over the best baseline (wins 19/23).</sub></td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%"><img src="docs/assets/figV2/05b.model_attribution_ism/fig_model_attribution_ism.png" width="100%" alt="In-silico mutagenesis attribution concentrating at minus-10 and plus-1"></td>
    <td width="50%"><img src="docs/assets/figV2/05c.sequence_motif/fig_sequence_motif.png" width="100%" alt="Two-sample sequence logo and cross-species conservation, including archaeal TATA-box"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Fig. S3 · Model attribution (ISM).</b> In‑silico mutagenesis of the deployed model concentrates importance at the −10 box and TSS (+1), tracking the data's discriminative positions (Pearson *r* = 0.82). Attribution is read from the foundation‑model experts and head, not the grammar PWM branch.</sub></td>
    <td align="center"><sub><b>Fig. S4 · Sequence motifs (data‑side).</b> Two‑sample sequence logo and positional information for <i>E. coli</i>, and conservation of promoter elements across all 23 prokaryotes — including the archaeal TATA‑box in <i>H. volcanii</i> and <i>T. kodakarensis</i>. Computed from sequences alone (no model).</sub></td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%"><img src="docs/assets/figV2/07.promoter_diversity/fig_promoter_diversity.png" width="100%" alt="Promoter diversity negatively correlated with separability and MCC"></td>
    <td width="50%"><img src="docs/assets/figV2/08.error_analysis/fig_error_analysis.png" width="100%" alt="Errors concentrate in high-diversity species; lowest risk-coverage AURC"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Fig. S5 · Promoter diversity.</b> More diverse promoter repertoires are harder: diversity is negatively correlated with both representation separability (Spearman ρ = −0.59) and ensemble MCC (ρ = −0.58), and the trend holds after controlling for sample size (partial ρ = −0.44).</sub></td>
    <td align="center"><sub><b>Fig. S6 · Error analysis.</b> Errors concentrate in a few high‑diversity species and hug the decision boundary; on selective prediction DeepPro‑v2 has the fewest confident errors (lowest risk–coverage AURC, 3.9), and difficulty rises with promoter positional entropy (ρ = 0.58).</sub></td>
  </tr>
</table>

</details>

---

## Overview

| view | backbone | pooled dim | source |
|------|----------|-----------|--------|
| 0 | DNABERT 3‑mer | 768 | `zhihan1996/DNA_bert_3` (HF) |
| 1 | DNABERT 6‑mer | 768 | `zhihan1996/DNA_bert_6` (HF) |
| 2 | DNABERT‑2 | 768 | `zhihan1996/DNABERT-2-117M` (HF) |
| 3 | Nucleotide Transformer v2 (50M) | 512 | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (HF) |
| 4 | ProkBERT‑mini‑c | 384 | `neuralbioinfo/prokbert-mini-c` (HF) |

The five view embeddings are combined by a **soft‑MoE fusion** (a learned
per‑species gate over views, initialised uniform ≡ plain mean fusion) plus a
differentiable **σ‑factor grammar** branch (−35 / spacer / −10 PWM scan) whose
mechanism vector is concatenated before the classifier head.

## Repository layout

```
DeepPro-v2/
├── README.md · requirements.txt
├── iPro/models/            # V1 backbone base (reused unchanged): DNAbert, DNAbert2, NTv2, FusionNet, deepPro
├── iProV2/
│   ├── models/             # deeppro_v2 · ProkBERT · moe_fusion · grammar_moe
│   ├── data/physics.py     # one-hot + SantaLucia ΔG helpers
│   ├── dataset.py          # 5-view Dataset (tokenizes all views up front)
│   └── infer.py            # ⭐ minimal inference entry point
├── docs/                   # self-contained showcase site → GitHub Pages (see below)
├── data/                   # ⬇ empty placeholder — download benchmark from Zenodo
└── weights/                # ⬇ empty placeholder — download checkpoints from Zenodo
```

> **Note**: DeepPro‑v2 builds on four V1 backbones, so the `iPro/` package is
> bundled here as a dependency. Keep the two‑package layout intact — the imports
> (`from iPro.models...`, `from iProV2.models...`) rely on it.

## Installation

```bash
# 1) create an environment (Python 3.10 recommended)
conda create -n deeppro python=3.10 -y
conda activate deeppro

# 2) install PyTorch matching your CUDA/CPU from https://pytorch.org
#    (e.g. a CUDA build; DeepPro-v2 runs on GPU by default, CPU also works)

# 3) install the rest
pip install -r requirements.txt
#    ProkBERT must be installed without deps (see requirements.txt):
pip install prokbert --no-deps
pip install biopython h5py
```

**First run downloads the backbones from HuggingFace** (DNABERT‑2, NT‑v2,
ProkBERT and DNABERT 3/6‑mer, with `trust_remote_code=True`); an internet
connection is required once, after which they are cached locally.

## Download data and weights

Both are hosted on Zenodo and extracted into the empty placeholders:

- **Dataset** → `data/` — see [`data/README.md`](data/README.md)
- **Weights** → `weights/` — see [`weights/README.md`](weights/README.md)
  (fold‑1 per species, ~35 GB)

## Run inference

```bash
cd iProV2
python infer.py --species 1            # one species
python infer.py --species 1,10,17      # several
python infer.py --species all          # all 23 species
```

For each species, `infer.py` loads the fold checkpoint(s), runs the test set
through the model, averages fold logits (ensemble), and prints
**ACC / Sn / Sp / Pre / F1 / MCC / AUC**; per‑sample predictions are saved to
`iProV2/infer_out/s{species}_pred.npz`. Point elsewhere with
`--data_dir` / `--weights_dir` if you extracted the archives outside the repo.

### Offline / custom backbone paths

If a machine is offline or you mirror DNABERT locally, override the DNABERT
k‑mer base directories via environment variables (default = HF hub ids):

```bash
export DNABERT_3MER=/path/to/DNA_bert_3
export DNABERT_6MER=/path/to/DNA_bert_6
```

The `docs/` folder is a **self‑contained static site** (HTML/CSS/JS with fonts
and figures bundled locally, plus a `.nojekyll` marker). Two ways to view it:

- **Locally**: open `docs/index.html` in a browser (works offline).
- **Online via GitHub Pages**: the folder is already laid out for Pages. In the
  repository, go to *Settings → Pages → Build and deployment → Source: Deploy
  from a branch → Branch: `main` / folder: `/docs`*. GitHub then serves the site
  at `https://<user>.github.io/<repo>/`; link that URL at the top of this README.

## Citation

```bibtex
@article{deeppro_v2,
  title   = {<TITLE>},
  author  = {<AUTHORS>},
  journal = {<JOURNAL>},
  year    = {<YEAR>},
  doi     = {<DOI>}
}
```

## License

`<LICENSE>` (TODO). Third‑party backbones (DNABERT, DNABERT‑2, Nucleotide
Transformer, ProkBERT) remain under their respective upstream licenses.
