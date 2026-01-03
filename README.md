# Tumor Drug Response Prediction - Multi-Modal Omics + Chemistry (TDRP)

## Overview
- Cleaned and aligned GDSC2 dose-response, cell-line metadata, RNA-seq expression, and drug SMILES into processed parquet tables plus tissue-aware splits (random, cell-line holdout, tissue holdout).
- Baseline PCA + MLP (sklearn) trained on concatenated omics/drug PCs; torch fusion model with optional VAE encoder for omics.
- Test set performance (PCA+MLP baseline): random RMSE 2.61 / r 0.31; cell-line holdout RMSE 2.71 / r 0.23; tissue holdout RMSE 2.77 / r 0.19.
- SHAP pipeline surfaces gene signatures/pathways driving sensitivity vs resistance per drug/tissue.
- Scope: RNA-seq expression only today (multi-omics branches are a planned extension).

Predict ln(IC50) for cell line-drug pairs by fusing gene expression with drug structure fingerprints, targeting generalization to unseen cell lines or tissues.

## Architecture
```mermaid
flowchart LR
  A[Raw GDSC2 files] --> B[EDA]
  A --> C[Preprocess + Align]
  C --> D[Processed parquet tables]
  D --> E[Coverage filtering]
  E --> F[Splits: random/cell-line/tissue]
  F --> G[Baselines: PCA+MLP]
  F --> H[Fusion model]
  G --> I[Metrics + plots]
  H --> I
  H --> J[SHAP (gene + FP)]
  G --> K[SHAP (PCA baseline)]
  H --> L[Embeddings]
```

## Modeling Details
- Fusion strategy: encode omics + drug fingerprints, concatenate latents, then regress with a small MLP (`src/tdrp/models/fusion.py`).
- VAE option: denoises omics before encoding; use when noise reduction helps generalization (validate via ablations).
- Drug featurization: Morgan fingerprints (RDKit). A graph featurizer stub exists for future graph-based encoders.
- Omics inputs: RNA-seq only. Multi-omics branches (mutation/CNV/methylation) are not implemented yet.

## Splits and Coverage Filtering
- Coverage thresholds (defaults): min drug coverage=0.7, min cell-line coverage=0.6, drop tissues with <15 cell lines.
- Random pair split: 70/15/15 train/val/test, tissue-stratified by default.
- Cell-line holdout: 70/15/15 train/val/test per tissue (cell lines are disjoint).
- Tissue holdout: test set = selected tissues; remaining tissues split 80/20 train/val.

## Data Leakage Considerations
- The current splits are tissue/cell-line aware but not temporal. If batch effects or time-based leakage are a concern, add a time- or batch-stratified split on the raw GDSC2 metadata.

## Data Sources (expected under `data/raw/`)
- `GDSC2_fitted_dose_response_27Oct23.xlsx` - fitted dose-response with ln(IC50).
- `Cell_Lines_Details.xlsx` - COSMIC metadata (tissue, site, cancer type).
- `screened_compounds_rel_8.5.csv` - drug IDs, targets, SMILES.
- `TableS1A.xlsx` or `rnaseq_*20191101*.{txt,csv}` - RNA-seq expression matrix.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# RDKit is best installed via conda:
conda install -c conda-forge rdkit
# Optional drug metadata enrichment (network calls to PubChem/ChEMBL):
pip install drugs==0.1.1
```

## Pipeline (reproducible commands)
### 1) EDA on raw GDSC2 files
```bash
python scripts/eda_gdsc2.py
```
Plots are written to `outputs/eda/` and key counts are printed.

### 2) Preprocess into parquet feature tables
```bash
python scripts/preprocess_gdsc.py \
  --raw-dir data/raw \
  --outdir data/processed \
  --n-genes 2000 \
  --fingerprint-bits 1024 \
  --manifest-out outputs/data_manifest.json
```
Outputs: `omics.parquet`, `drug_fingerprints.parquet`, `labels.parquet`, `metadata.parquet`, plus `outputs/data_manifest.json`.

### 3) Build filtered train/val/test splits
```bash
python scripts/make_splits.py \
  --config configs/default.yaml \
  --outdir data/processed/splits \
  --test-tissues lung_NSCLC urogenital_system leukemia aero_dig_tract breast
```
Emits `random_pair_split.csv`, `cellline_holdout_split.csv`, `tissue_holdout_split.csv` with coverage filtering.

### 4) (Optional) Refresh data manifest with split sizes
```bash
python scripts/data_manifest.py \
  --processed-dir data/processed \
  --splits-dir data/processed/splits \
  --out outputs/data_manifest.json
```

### 5) (Optional) Enrich drug metadata (PubChem/ChEMBL)
```bash
python scripts/enrich_drugs.py \
  --processed-dir data/processed \
  --sleep-sec 0.2
```
Outputs `data/processed/drug_metadata_enriched.parquet` (plus per-drug markdown reports if
`--report-dir` is provided).

### 6) Train models
- Torch fusion model:
```bash
python scripts/train.py --config configs/default.yaml --output outputs/gdsc2_run1
```
(`split_strategy`/`use_vae` configurable in YAML.)

- PCA+MLP baseline on a chosen split:
```bash
python scripts/train_baseline_ml.py \
  --processed-dir data/processed \
  --split-csv data/processed/splits/random_pair_split.csv \
  --outdir outputs/baseline_random \
  --omics-pca 256 --drug-pca 128 --hidden-layers 512,256
```

### 7) SHAP explanations for the torch model
```bash
python scripts/explain.py \
  --config configs/default.yaml \
  --checkpoint outputs/gdsc2_run1/model.pt \
  --output outputs/gdsc2_run1/shap_values.npz \
  --sample-size 500 \
  --background-size 100
```
The exported `.npz` can be summarized to highlight gene signatures/pathways influencing sensitivity/resistance.

### Optional: Embed/visualize representations
```bash
python scripts/plot_embeddings.py \
  --embeddings outputs/gdsc2_run1/cell_embeddings.npz \
  --metadata data/processed/metadata.parquet \
  --method pca --color-by tissue \
  --outpath outputs/gdsc2_run1/embeddings_pca_tissue.png
```

## Results (PCA+MLP baseline, test split metrics)
| Split | Test RMSE | Test Pearson r |
| --- | --- | --- |
| Random | 2.61 | 0.31 |
| Cell-line holdout | 2.71 | 0.23 |
| Tissue holdout | 2.77 | 0.19 |

Baseline reference and plots live in `reports/baseline_ml.md`.

## Baselines and Reporting
- `scripts/train_baseline_ml.py` is the reference sklearn baseline.
- `scripts/train_torch_baseline.py` mirrors PCA + MLP in torch for comparability.
- `scripts/train.py` trains the fusion model with optional VAE.

## SHAP Interpretation
- Fusion model SHAP uses gene symbols and fingerprint bit columns as feature names.
- PCA baseline SHAP uses PCA component names; for gene-level interpretation, prefer the fusion model.
- Optional: pass `--manifest outputs/data_manifest.json` to `scripts/explain_torch_baseline.py` to attach original feature name lists.

## Reproducibility Notes
- Dependency versions are pinned in `requirements.txt` and `pyproject.toml`.
- Random seeds are set in training/explanation scripts; for strict determinism, also set `PYTHONHASHSEED=42`.
- GPU is recommended for SHAP and torch training; CPU runs are supported but slower.

## Outputs Organization
- `outputs/eda/` is generated by `scripts/eda_gdsc2.py`.
- `data/processed/eda/` contains curated EDA artifacts (examples/checkpoints).
- `outputs/data_manifest.json` summarizes processed data, missingness, and split sizes (if splits exist).

## Extending
- Swap PCA encoder for VAE in the torch model (`model.use_vae=true`, `model.omics_latent_dim` tuned per config).
- Replace Morgan fingerprints with a graph featurizer (`src/tdrp/featurizers/drugs.py` scaffold) to capture substructures.
- Stress-test generalization with `split_strategy=tissue_holdout` or custom `--test-tissues` in `make_splits.py`.
- Add richer interpretability: SHAP per drug/tissue, pathway enrichment on top genes, and embedding clustering vs tissues.
