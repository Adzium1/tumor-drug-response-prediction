# Tumor Drug Response Prediction - Multi-Modal Omics + Chemistry (TDRP)

Predict ln(IC50) for cell line - drug pairs by fusing gene expression with drug structure fingerprints. The goal is to generalize to unseen cell lines or tissues by combining omics and chemistry.

## Data expectations
- `data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx`
- `data/raw/Cell_Lines_Details.xlsx`
- `data/raw/screened_compounds_rel_8.5.csv`
- `data/raw/TableS1A.xlsx` (expression matrix)

## Project layout
```
configs/default.yaml
data/{raw,processed,external}/
scripts/{preprocess_gdsc.py,train.py,explain.py,eda_gdsc2.py}
src/tdrp/...  # config, utils, data, featurizers, models, training, analysis
requirements.txt
pyproject.toml
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# RDKit is best installed via conda:
conda install -c conda-forge rdkit
```

## Usage
### EDA on raw GDSC2 files
```bash
python scripts/eda_gdsc2.py
```

### Preprocess GDSC2 into parquet tables
```bash
python scripts/preprocess_gdsc.py \
  --raw-dir data/raw \
  --outdir data/processed \
  --n-genes 2000 \
  --fingerprint-bits 1024
```

### Train
```bash
python scripts/train.py --config configs/default.yaml --output outputs/gdsc2_run1
```

### Explain with SHAP
```bash
python scripts/explain.py \
  --config configs/default.yaml \
  --checkpoint outputs/gdsc2_run1/model.pt \
  --output outputs/gdsc2_run1/shap_values.npz \
  --sample-size 500 \
  --background-size 100
```

## Extending
- Switch between PCA/MLP and VAE encoders via config (`model.use_vae`, latent dims).
- Replace Morgan fingerprints with a graph featurizer (stub in `tdrp/featurizers/drugs.py`).
- Evaluate generalization with `split_strategy=tissue_holdout` to hold out specific tissues.
