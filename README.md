# Tumor Drug Response Prediction — Multi-Modal Omics + Chemistry

Scaffold for predicting cell line drug sensitivity (ln(IC50)) by fusing transcriptomic profiles with chemical structure. The code mirrors the lab-grade workflow used by Halin to bridge RNA-seq and Mass Spec pipelines: careful data curation, dimensionality reduction (PCA/VAE), two-stream neural encoders, stringent splits (leave-cell-line-out / tissue holdout), and SHAP-based biological interpretation.

## Scientific Aim
- Hypothesis: a dual-branch model (omics + drug) captures non-linear gene–drug interactions that linear baselines miss.
- Inputs: cell line expression matrix (20k genes → reduced to 1–3k), drug SMILES → Morgan fingerprints (1024–2048 bits) or molecular graphs.
- Output: ln(IC50) (or AUC) per cell line–drug pair.
- Interpretability deliverable: SHAP attribution linking drug response to gene expression patterns (e.g., EGFR high expression driving sensitivity).

## Repository Layout
- `configs/` — experiment configs (`default.yaml` mirrors the dataclasses in `tdrp.config`).
- `data/` — `raw/`, `processed/`, `external/` placeholders (never tracked).
- `scripts/` — CLI utilities: preprocessing, training, SHAP export.
- `src/tdrp/` — library code.
  - `config.py` — YAML ↔ dataclass bridge.
  - `data/` — preprocessing, alignment, and split strategies (leave-cell-line-out, tissue holdout, k-fold).
  - `featurizers/` — omics dimensionality reduction (PCA) and SMILES featurization (fingerprints, graphs).
  - `models/` — omics/drug encoders, optional VAE denoiser, fusion regressor, losses.
  - `training/` — trainer loop, metrics (RMSE, Pearson).
  - `analysis/` — SHAP wrapper for post-hoc biological insight.
  - `utils/` — logging, seeding, I/O helpers.

## Environment
Install dependencies (RDKit via conda is recommended for stability):
```bash
pip install -r requirements.txt
# or
conda install -c conda-forge rdkit
```

## Data Expectations (GDSC/CCLE style)
- Expression table (`csv`/`parquet`): columns `cell_line`, gene columns (TPM/log2). Preprocessing selects top-variance genes (default 2000) and z-scores per gene.
- Drug table: columns `drug`, `smiles`.
- Labels: columns `cell_line`, `drug`, `ln_ic50` (or `auc` → rename to `ln_ic50` before running).
- Optional metadata: columns `cell_line`, `tissue` for tissue-specific holdout.
- Processed outputs (parquet): `omics.parquet`, `drug_fingerprints.parquet`, `labels.parquet`, `metadata.parquet` (optional).

## Workflow
1) **Preprocess** (feature selection + fingerprints + alignment)
```bash
python scripts/preprocess_gdsc.py \
  --expression data/raw/expression.csv \
  --labels data/raw/labels.csv \
  --drug-smiles data/raw/drugs.csv \
  --metadata data/raw/metadata.csv \
  --outdir data/processed \
  --n-genes 2000 --fingerprint-bits 1024
```
Outputs stored in `data/processed/`.

2) **Train** (two-branch network; default is leave-cell-line-out)
```bash
python scripts/train.py --config configs/default.yaml --output outputs
```
- Model: omics encoder (MLP or VAE latent), drug encoder (MLP on fingerprints), fusion MLP → ln(IC50).
- Splits:
  - `leave_cell_line_out`: deterministic 20% cell-line holdout for generalization to unseen lines.
  - `tissue_holdout`: set `training.split_strategy=tissue_holdout` and `training.tissue_holdout=<TISSUE>` in YAML.
  - `kfold`: cell-line level k-fold for robustness checks.
- Metrics: RMSE + Pearson on held-out cell lines.

3) **Interpretability (SHAP)** — gene/drug contributions per prediction
```bash
python scripts/explain.py --config configs/default.yaml --checkpoint outputs/model.pt --output outputs/shap_values.npz
```
Produces SHAP values for combined omics + drug features. Visualize with your plotting tool of choice (e.g., `shap.summary_plot` using saved arrays).

## Extending the Research
- Swap PCA with the built-in VAE branch (`model.use_vae=true`) to denoise high-dimensional omics (parallels Mass Spec latent denoising).
- Replace fingerprints with molecular graphs + GNN encoder (hooks provided in `featurizers/drugs.py`).
- Tissue-specific blind tests: train on lung/breast, hold out melanoma via `training.tissue_holdout`.
- Biomarker discovery: rank SHAP attributions per drug to surface resistance/sensitivity markers (e.g., BRAF for vemurafenib).

## Reproducibility
- Seeding handled in `tdrp.utils.seed.set_seed`.
- Config-driven runs: `configs/default.yaml` is aligned with `tdrp.config` dataclasses for experiment tracking.
- Logging: minimal console logging; extend via `tdrp.utils.logging.setup_logging`.

## Why This Matches the Brief
- Two-stream architecture (omics + drug) with latent fusion and dropout regularization.
- Explicit dimensionality reduction path (PCA/optional VAE) to mirror Mass Spec denoising requirements.
- Rigorous splits (leave-cell-line-out, tissue holdout) aligned with translational generalization.
- Interpretability baked in (SHAP) to justify predictions biologically.
- Clear data curation steps (gene variance filtering, z-scoring, fingerprint computation, alignment of pairs).
