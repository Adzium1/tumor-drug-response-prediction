"""
Quick EDA for GDSC2 raw files.

Run: python scripts/eda_gdsc2.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tdrp.utils.io import ensure_dir

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


RAW_DIR = Path("data/raw")
EDA_OUT = Path("outputs/eda")


def _normalize_col(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _find_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> Optional[str]:
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    logger.warning("Expected column %s in %s, got: %s", list(candidates), context, list(df.columns))
    return None


def _assert_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> str:
    col = _find_column(df, candidates, context)
    assert col is not None, f"Expected column {list(candidates)} in {context}, got: {list(df.columns)}"
    return col


def _print_basic_info(name: str, df: pd.DataFrame, key_cols: Iterable[str]) -> None:
    print(f"\n{name}: shape={df.shape}")
    print(f"{name} columns: {list(df.columns)}")
    print(df.head())
    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"Missing in {col}: {missing}")


def _save_hist(series: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(series.dropna(), bins=50, kde=False)
    plt.title(title)
    plt.xlabel(series.name or title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_bar_counts(counts: pd.Series, title: str, path: Path, top_n: int = 20) -> None:
    plt.figure(figsize=(8, 5))
    counts.sort_values(ascending=False).head(top_n).plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_dir(EDA_OUT)

    dose_path = RAW_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx"
    cells_path = RAW_DIR / "Cell_Lines_Details.xlsx"
    drugs_path = RAW_DIR / "screened_compounds_rel_8.5.csv"
    expr_path = RAW_DIR / "TableS1A.xlsx"

    dose_df = pd.read_excel(dose_path, engine="openpyxl")
    cells_df = pd.read_excel(cells_path, engine="openpyxl")
    drugs_df = pd.read_csv(drugs_path)
    expr_df = pd.read_excel(expr_path, engine="openpyxl")

    cosmic_col = _assert_column(dose_df, ["cosmic_id", "cosmic", "cosmic_identifier"], "dose-response")
    drug_name_col = _assert_column(dose_df, ["drug_name"], "dose-response")
    ln_ic50_col = _find_column(
        dose_df,
        ["ln_ic50", "ln_ic50_um", "ln_ic50_(um)", "ln_ic50_umol", "ln_ic50_(uM)".lower()],
        "dose-response",
    )
    ic50_col = _find_column(dose_df, ["ic50", "ic50_(um)", "ic50_um"], "dose-response")
    _print_basic_info("Dose-response", dose_df, [cosmic_col, drug_name_col, ln_ic50_col or ic50_col])

    print("\nTop drugs by measurement count:")
    print(dose_df[drug_name_col].value_counts().head(10))
    print("\nTop cell lines by measurement count:")
    cell_line_col = _find_column(dose_df, ["cell_line_name", "cell_line", "line"], "dose-response")
    if cell_line_col:
        print(dose_df[cell_line_col].value_counts().head(10))

    if ln_ic50_col is not None:
        _save_hist(dose_df[ln_ic50_col], "LN_IC50 distribution", EDA_OUT / "ln_ic50_hist.png")
    elif ic50_col is not None:
        ln_vals = np.log(pd.to_numeric(dose_df[ic50_col], errors="coerce"))
        _save_hist(ln_vals, "log(IC50) distribution", EDA_OUT / "ln_ic50_hist.png")

    drug_id_col = _assert_column(drugs_df, ["drug_id"], "screened compounds")
    print("\nDrug table unique drugs:", drugs_df[drug_id_col].nunique())
    target_col = _find_column(drugs_df, ["target", "putative_target", "target_pathway"], "screened compounds")
    if target_col:
        print("Missing targets:", drugs_df[target_col].isna().sum())

    cosmic_meta_col = _assert_column(cells_df, ["cosmic_identifier", "cosmic_id", "cosmic"], "cell lines")
    line_meta_col = _assert_column(cells_df, ["cell_line_name", "line", "cell_line", "sample_name"], "cell lines")
    tissue_meta_col = _find_column(cells_df, ["gdsc_tissue_descriptor_1", "tissue"], "cell lines")
    print("\nCell line metadata unique COSMIC IDs:", cells_df[cosmic_meta_col].nunique())
    print("Cell line metadata unique lines:", cells_df[line_meta_col].nunique())

    # Expression orientation inspection
    print("\nExpression matrix shape:", expr_df.shape)
    print("First few column labels:", list(expr_df.columns[:5]))
    print("First few row labels:", expr_df.head()[expr_df.columns[0]].head())
    numeric_columns = {c for c in expr_df.columns if str(c).isdigit()}
    print("Number of numeric column headers (possible COSMIC IDs):", len(numeric_columns))

    # Overlaps between expression and metadata
    cosmic_series = pd.to_numeric(cells_df[cosmic_meta_col], errors="coerce").dropna()
    cosmic_ids = set(str(int(x)) for x in cosmic_series)
    expr_columns_as_ids = {str(int(c)) for c in numeric_columns if str(c).isdigit()}
    overlap_cols = cosmic_ids & expr_columns_as_ids
    print("Overlap between COSMIC IDs (metadata) and expression columns:", len(overlap_cols))

    # Measurements per drug and per tissue
    measurements_per_drug = dose_df[drug_name_col].value_counts()
    _save_bar_counts(measurements_per_drug, "Measurements per drug (top 20)", EDA_OUT / "measurements_per_drug.png")

    if tissue_meta_col:
        meta_tissue = cells_df[[cosmic_meta_col, tissue_meta_col]]
        merged = dose_df.merge(meta_tissue, left_on=cosmic_col, right_on=cosmic_meta_col, how="left")
        if tissue_meta_col in merged:
            _save_bar_counts(
                merged[tissue_meta_col].value_counts(),
                "Measurements per tissue (top 20)",
                EDA_OUT / "measurements_per_tissue.png",
            )

    print(f"\nEDA complete. Plots saved to {EDA_OUT}")


if __name__ == "__main__":
    main()
