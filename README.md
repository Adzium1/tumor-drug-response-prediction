# Tumor Drug Response Prediction - Multi-Modal Omics + Chemistry (TDRP)

Predict ln(IC50) for cell line - drug pairs by fusing gene expression with drug structure fingerprints. The working hypothesis: combining denoised omics representations with chemical fingerprints improves generalization to unseen cell lines or tissues.

## Data expectations
- `expression.csv`: `cell_line`, gene1, gene2, ..., geneN
- `labels.csv`: `cell_line`, `drug`, `ln_ic50`
- `drugs.csv`: `drug`, `smiles`
- `metadata.csv` (optional): `cell_line`, `tissue` (and other annotations)

## Project layout
```
tdrp/
  configs/default.yaml
  data/{raw,processed,external}/
  scripts/{preprocess_gdsc.py,train.py,explain.py}
  src/tdrp/...  # config, utils, data, featurizers, models, training, analysis
  requirements.txt
  pyproject.toml
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r tdrp/requirements.txt
# RDKit is best installed via conda:
conda install -c conda-forge rdkit
```

## Usage
### Preprocess GDSC-style tables
```bash
python tdrp/scripts/preprocess_gdsc.py \
  --expression tdrp/data/raw/expression.csv \
  --labels tdrp/data/raw/labels.csv \
  --drug-smiles tdrp/data/raw/drugs.csv \
  --metadata tdrp/data/raw/metadata.csv \
  --outdir tdrp/data/processed \
  --n-genes 2000 \
  --fingerprint-bits 1024
```

### Train
```bash
python tdrp/scripts/train.py --config tdrp/configs/default.yaml --output tdrp/outputs
```

### Explain with SHAP
```bash
python tdrp/scripts/explain.py \
  --config tdrp/configs/default.yaml \
  --checkpoint tdrp/outputs/model_fold0.pt \
  --output tdrp/outputs/shap_values.npz \
  --sample-size 500 \
  --background-size 100
```

## Extending
- Swap PCA or MLP encoders with the VAE by toggling `model.use_vae` or adjusting latent dims in config.
- Replace Morgan fingerprints with a graph featurizer (stub in `tdrp/featurizers/drugs.py`).
- Evaluate generalization with `split_strategy=tissue_holdout` to hold out specific tissues.
