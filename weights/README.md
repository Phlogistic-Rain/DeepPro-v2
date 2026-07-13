# Model weights (download separately)

This folder is an **empty placeholder**. The trained DeepPro‑v2 checkpoints are
hosted on Zenodo (large binaries) and are **not** shipped in the GitHub
repository.

## What is published

The released subset contains **one fold (fold‑1) per species**, i.e. 23
checkpoints (~35 GB total; each checkpoint is a full DeepPro‑v2 `state_dict`
≈ 1.5 GB: all five fine‑tuned backbones + fusion + grammar branch).

> The full 5‑fold × 23‑species ensemble used for the headline number is ~162 GB
> and is available on request; fold‑1 reproduces the single‑model test metrics.

## Download

1. Get the weights archive from the Zenodo record:
   **`<ZENODO_DOI>`** (TODO: fill in the DOI/URL after minting the record).
2. Extract it **here**, so the layout becomes:

```
weights/
├── s1/1/checkpoints/DeepPro_epoch*_mcc*.pth
├── s2/1/checkpoints/DeepPro_epoch*_mcc*.pth
└── ... s23/1/checkpoints/DeepPro_epoch*_mcc*.pth
```

`iProV2/infer.py` auto‑discovers checkpoints under `weights/s{species}/...`
(it also accepts flatter layouts such as `weights/s1/*.pth`). If several folds
are present for a species, their logits are averaged (ensemble).

You do **not** need the stage‑0 pretrained backbones for inference — a released
checkpoint already contains every weight; the HuggingFace base models are only
loaded to build the architecture and are immediately overwritten.
