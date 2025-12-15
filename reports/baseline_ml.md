Baseline PCA+MLP (sklearn) – ln(IC50)
======================================

Setup
- Data: GDSC2 processed tables (941 cell lines, 295 drugs, 236,792 pairs) filtered at drug frac ≥0.7, cell frac ≥0.6; splits from `scripts/make_splits.py`.
- Model: PCA (omics 256 comps, drug FP 128 comps) → MLPRegressor (hidden layers 512, 256, ReLU, early stopping, max_iter=50).
- Splits evaluated: random pair, cell-line holdout, tissue holdout.

Results (test)
- random: RMSE 2.608, Pearson r 0.309
- cell_holdout: RMSE 2.709, Pearson r 0.230
- tissue_holdout: RMSE 2.765, Pearson r 0.193

Key plots (see outputs/)
- `outputs/baseline_random/pred_vs_true_test.png`, `residuals_test.png`
- `outputs/baseline_cellhold/pred_vs_true_test.png`, `residuals_test.png`
- `outputs/baseline_tissuehold/pred_vs_true_test.png`, `residuals_test.png`
- Per-drug/tissue top-r barplots in the same folders.

Interpretation
- Random split is easiest; cell- and tissue-holdout degrade as expected, showing the model uses cell-line identity structure.
- Modest signal (r ≈0.19–0.31) with limited overfitting (train vs test RMSE close); drug FPs are partly zero due to missing SMILES, limiting ceiling.
- This PCA+MLP serves as a baseline: torch models with learned encoders should aim to beat these RMSE/r numbers and show at least comparable tissue clustering in embedding projections.
