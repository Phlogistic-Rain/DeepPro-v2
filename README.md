# DeepPro‑v2 — Multi‑view, multi‑backbone prokaryotic promoter prediction

DeepPro‑v2 predicts prokaryotic promoters from 81‑bp DNA sequences by
aggregating **five** genomic language‑model backbones through a
soft mixture‑of‑experts fusion, with an interpretable σ‑factor *grammar* branch.

- 📄 **Paper**: `<PAPER_URL>` (TODO)
- 💾 **Data & weights (Zenodo)**: `<ZENODO_DOI>` (TODO)
- 🌐 **Interactive demo**: https://phlogistic-rain.github.io/DeepPro-v2/ (see [Interactive demo](#interactive-demo))

> This repository is a **minimal, inference‑only** release: model code, dataset
> code, and a single‑command inference script that reproduces the reported test
> metrics from the published weights. Analysis/experiment scripts are not
> included.

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
