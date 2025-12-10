"""
Quick EDA for GDSC2 raw files.

Run: python scripts/eda_gdsc2.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.utils.io import ensure_dir

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


RAW_DIR = Path("data/raw")
EDA_OUT = Path("outputs/eda")


def _normalize_col(col: str) -> str:
    return str(col).strip().lower().replace("\n", "_").replace(" ", "_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    return df


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

    dose_df = pd.read_excel(dose_path, engine="openpyxl")
    cells_df = _normalize_columns(pd.read_excel(cells_path, engine="openpyxl"))
    drugs_df = pd.read_csv(drugs_path)

    cosmic_col = _assert_column(dose_df, ["cosmic_id", "cosmic", "cosmic_identifier"], "dose-response")
    sanger_col = _assert_column(dose_df, ["sanger_model_id"], "dose-response")
    drug_name_col = _assert_column(dose_df, ["drug_name"], "dose-response")
    ln_ic50_col = _find_column(
        dose_df,
        ["ln_ic50", "ln_ic50_um", "ln_ic50_(um)", "ln_ic50_umol", "ln_ic50_(umol)"],
        "dose-response",
    )
    ic50_col = _find_column(dose_df, ["ic50", "ic50_(um)", "ic50_um"], "dose-response")
    _print_basic_info("Dose-response", dose_df, [cosmic_col, sanger_col, drug_name_col, ln_ic50_col or ic50_col])

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
    line_meta_col = _assert_column(cells_df, ["sample_name", "cell_line_name", "line", "cell_line"], "cell lines")
    tissue_meta_col = _find_column(cells_df, ["gdsc_tissue_descriptor_1", "tissue"], "cell lines")
    print("\nCell line metadata unique COSMIC IDs:", cells_df[cosmic_meta_col].nunique())
    print("Cell line metadata unique lines:", cells_df[line_meta_col].nunique())
    if tissue_meta_col:
        print("Unique tissues (first 10):", cells_df[tissue_meta_col].dropna().unique()[:10])

    # RNA-seq expression loading
    rnaseq_dir = RAW_DIR
    expr_files = sorted(
        list(rnaseq_dir.glob("rnaseq_fpkm_20191101.*"))
        + list(rnaseq_dir.glob("rnaseq_read_count_20191101.*"))
        + list(rnaseq_dir.glob("rnaseq_*20191101*.txt"))
        + list(rnaseq_dir.glob("rnaseq_*20191101*.csv"))
    )
    if not expr_files:
        print(f"No RNA-seq file found under {rnaseq_dir}.")
    else:
        fpkm_files = [p for p in expr_files if "fpkm" in p.name.lower()]
        expr_file = fpkm_files[0] if fpkm_files else expr_files[0]
        sep = "\t" if expr_file.suffix == ".txt" else ","
        expr_df = pd.read_csv(expr_file, sep=sep, low_memory=False)
        print(f"\nExpression file: {expr_file.name}, shape={expr_df.shape}")
        print("First columns:", list(expr_df.columns[:5]))
        print("Head:\n", expr_df.head())

        expr_cols = set(expr_df.columns[2:])  # sample IDs start after gene_id/symbol columns
        sanger_ids = set(dose_df[sanger_col].astype(str))
        cosmic_dose = set(str(int(x)) for x in pd.to_numeric(dose_df[cosmic_col], errors="coerce").dropna())
        cosmic_meta = set(str(int(x)) for x in pd.to_numeric(cells_df[cosmic_meta_col], errors="coerce").dropna())
        overlap_sanger = sanger_ids & expr_cols
        print(f"Overlap (SANGER_MODEL_ID vs expression columns): {len(overlap_sanger)}")
        print(f"Overlap (dose COSMIC vs expression columns): 0")
        print(f"Overlap (metadata COSMIC vs expression columns): 0")

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
