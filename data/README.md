# Benchmark dataset (download separately)

This folder is an **empty placeholder**. The prokaryotic promoter benchmark
(23 species, 81‑bp sequences) is hosted on Zenodo and is **not** shipped in the
GitHub repository.

## Download

1. Get the dataset archive from the Zenodo record:
   **https://doi.org/10.5281/zenodo.21344602** (DOI: `10.5281/zenodo.21344602`).
2. Extract it **here**, so the layout becomes:

```
data/
└── Benchmark Dataset/
    └── csv/
        ├── Test/
        │   ├── 1_test.csv            # species 1 test set  ← used by infer.py
        │   ├── 2_test.csv
        │   └── ... 23_test.csv
        └── Train_5fold/              # (training only; not needed for inference)
            └── {species}/fold_{fold}_{train,val}.csv
```

Every CSV has two columns: **`text`** (an 81‑bp DNA sequence) and **`label`**
(`1` = promoter, `0` = non‑promoter).

## What inference needs

`iProV2/infer.py` only reads `Benchmark Dataset/csv/Test/{species}_test.csv`.
The `Train_5fold/` split is included in the same archive for completeness /
retraining but is not required to reproduce the reported test metrics.
