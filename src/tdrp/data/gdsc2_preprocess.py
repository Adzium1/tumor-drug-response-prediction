from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from tdrp.featurizers.drugs import featurize_drug_table
from tdrp.utils.io import ensure_dir, save_parquet

logger = logging.getLogger(__name__)


def _normalize_col(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _normalize_id(value) -> Optional[str]:
    if pd.isna(value):
        return None
    try:
        return str(int(float(str(value))))
    except Exception:
        return str(value).strip()


def _get_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> str:
    candidates_norm = {c.lower(): c for c in candidates}
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in candidates_norm:
        if cand in normalized:
            return normalized[cand]
    raise ValueError(f"Expected column in {context}: one of {list(candidates_norm.values())}; got {list(df.columns)}")


def _get_optional_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    return None


def load_gdsc2_fitted(path: str) -> pd.DataFrame:
    """Load the fitted dose-response Excel file."""
    return pd.read_excel(path, engine="openpyxl")


def load_cell_line_details(path: str) -> pd.DataFrame:
    """Load cell line metadata details."""
    return pd.read_excel(path, engine="openpyxl")


def load_screened_compounds(path: str) -> pd.DataFrame:
    """Load screened compound table."""
    return pd.read_csv(path)


def load_expression_table(path: str) -> pd.DataFrame:
    """Load expression matrix."""
    return pd.read_excel(path, engine="openpyxl")


def build_metadata(cell_lines_df: pd.DataFrame) -> pd.DataFrame:
    """Construct metadata table with cell_line names and tissues."""
    cosmic_col = _get_column(cell_lines_df, ["cosmic_identifier", "cosmic_id", "cosmic"], "cell line details")
    line_col = _get_column(
        cell_lines_df,
        ["cell_line_name", "line", "cell_line", "sample_name", "sample"],
        "cell line details",
    )
    tissue_col = _get_column(
        cell_lines_df,
        ["gdsc_tissue_descriptor_1", "tissue", "tissue_type"],
        "cell line details",
    )
    site_col = _get_optional_column(cell_lines_df, ["site", "primary_site"])
    cancer_col = _get_optional_column(cell_lines_df, ["cancer_type", "cancer_type_detailed"])

    metadata = pd.DataFrame(
        {
            "cosmic_id": pd.to_numeric(cell_lines_df[cosmic_col], errors="coerce").astype("Int64"),
            "cell_line": cell_lines_df[line_col].astype(str),
            "tissue": cell_lines_df[tissue_col].astype(str),
        }
    )
    if site_col:
        metadata["site"] = cell_lines_df[site_col].astype(str)
    if cancer_col:
        metadata["cancer_type"] = cell_lines_df[cancer_col].astype(str)

    metadata = metadata.dropna(subset=["cosmic_id", "cell_line"])
    metadata = metadata.drop_duplicates(subset=["cosmic_id"])
    metadata = metadata.reset_index(drop=True)
    return metadata


def build_drug_table(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """Construct minimal drug table with IDs, names, targets, and placeholder SMILES."""
    drug_id_col = _get_column(compounds_df, ["drug_id"], "screened compounds")
    drug_name_col = _get_optional_column(compounds_df, ["drug_name", "drug", "compound"])
    target_col = _get_optional_column(compounds_df, ["target", "putative_target", "target_pathway"])
    smiles_col = _get_optional_column(compounds_df, ["smiles", "canonical_smiles"])

    drugs = pd.DataFrame({"drug": compounds_df[drug_id_col].astype(str)})
    if drug_name_col:
        drugs["drug_name"] = compounds_df[drug_name_col]
    if target_col:
        drugs["target"] = compounds_df[target_col]
    drugs["smiles"] = compounds_df[smiles_col] if smiles_col else np.nan

    drugs = drugs.drop_duplicates(subset=["drug"]).reset_index(drop=True)
    return drugs


def build_labels(gdsc2_df: pd.DataFrame, metadata: pd.DataFrame, drugs_df: pd.DataFrame) -> pd.DataFrame:
    """Build labels table: cell_line, drug, ln_ic50."""
    cosmic_col = _get_column(gdsc2_df, ["cosmic_id", "cosmic", "cosmic_identifier"], "dose-response")
    drug_id_col = _get_column(gdsc2_df, ["drug_id"], "dose-response")
    drug_name_col = _get_optional_column(gdsc2_df, ["drug_name"])

    ln_ic50_col = _get_optional_column(
        gdsc2_df,
        ["ln_ic50", "ln_ic50_um", "ln_ic50_(um)", "ln_ic50_umol", "ln_ic50_uM".lower(), "ln_ic50_(uM)".lower()],
    )
    ic50_col = _get_optional_column(gdsc2_df, ["ic50", "ic50_um", "ic50_(um)", "ic50_umol"])
    if ln_ic50_col is None and ic50_col is None:
        raise ValueError(
            f"Expected LN_IC50 or IC50 column in dose-response file; got {list(gdsc2_df.columns)}"
        )

    base = gdsc2_df[[cosmic_col, drug_id_col]].copy()
    if ln_ic50_col:
        base["ln_ic50"] = pd.to_numeric(gdsc2_df[ln_ic50_col], errors="coerce")
    else:
        ic50 = pd.to_numeric(gdsc2_df[ic50_col], errors="coerce")
        ic50 = ic50.replace({0: np.nan})
        base["ln_ic50"] = np.log(ic50)

    cosmic_map = {str(int(c)): name for c, name in zip(metadata["cosmic_id"], metadata["cell_line"])}
    base["cell_line"] = base[cosmic_col].apply(lambda x: cosmic_map.get(_normalize_id(x)))
    base["drug"] = base[drug_id_col].astype(str)
    if drug_name_col and "drug_name" not in drugs_df.columns:
        # add drug names if missing from drug table
        name_map = dict(zip(gdsc2_df[drug_id_col].astype(str), gdsc2_df[drug_name_col]))
        drugs_df["drug_name"] = drugs_df["drug"].map(name_map)

    labels = base.dropna(subset=["cell_line", "drug", "ln_ic50"]).reset_index(drop=True)
    labels = labels[np.isfinite(labels["ln_ic50"])]
    labels = labels[["cell_line", "drug", "ln_ic50"]]
    return labels.reset_index(drop=True)


def _detect_expression_orientation(df: pd.DataFrame) -> Tuple[str, str]:
    first_col = df.columns[0]
    normalized_first = _normalize_col(first_col)
    numeric_cols = [c for c in df.columns if str(c).isdigit() or "cosmic" in _normalize_col(c)]
    if normalized_first.startswith("cosmic") or str(first_col).isdigit():
        return "cells_rows", first_col
    if len(numeric_cols) >= max(1, len(df.columns) - 1):
        return "genes_rows", first_col
    gene_like = {"gene", "genes", "gene_symbol", "gene_symbols"}
    if normalized_first in gene_like:
        return "genes_rows", first_col
    return "genes_rows", first_col


def build_expression(expression_df: pd.DataFrame, metadata: pd.DataFrame, n_genes: int = 2000) -> pd.DataFrame:
    """Build expression matrix aligned to cell_line names, z-scored, top-variance genes."""
    orientation, id_col = _detect_expression_orientation(expression_df)
    logger.info("Detected expression orientation: %s (id column: %s)", orientation, id_col)

    if orientation == "genes_rows":
        gene_col = id_col
        expr = expression_df.set_index(gene_col)
        expr_numeric = expr.apply(pd.to_numeric, errors="coerce")
        expr_numeric = expr_numeric.transpose()
        expr_numeric.index.name = "sample_id"
        expr_numeric = expr_numeric.reset_index()
    else:
        sample_col = id_col
        expr_numeric = expression_df.copy()
        expr_numeric[sample_col] = expr_numeric[sample_col].apply(_normalize_id)

    sample_col = "sample_id" if "sample_id" in expr_numeric.columns else expr_numeric.columns[0]
    feature_cols = [c for c in expr_numeric.columns if c != sample_col]

    expr_numeric[feature_cols] = expr_numeric[feature_cols].apply(pd.to_numeric, errors="coerce")
    expr_numeric = expr_numeric.dropna(axis=1, how="all")

    cosmic_map = {str(int(c)): name for c, name in zip(metadata["cosmic_id"], metadata["cell_line"])}
    cell_line_names = set(metadata["cell_line"])

    def _map_sample(sample: str) -> Optional[str]:
        norm = _normalize_id(sample)
        if norm and norm in cosmic_map:
            return cosmic_map[norm]
        if sample in cell_line_names:
            return sample
        return None

    expr_numeric["cell_line"] = expr_numeric[sample_col].apply(_map_sample)
    expr_numeric = expr_numeric.dropna(subset=["cell_line"])
    expr_numeric = expr_numeric.drop(columns=[sample_col])
    expr_numeric = expr_numeric.set_index("cell_line")

    variances = expr_numeric.var(axis=0, ddof=0)
    top_genes = variances.sort_values(ascending=False).head(min(n_genes, len(variances))).index
    filtered = expr_numeric[top_genes]

    filtered = (filtered - filtered.mean()) / filtered.std(ddof=0)
    filtered = filtered.astype(np.float32)
    filtered = filtered.reset_index()
    return filtered


def align_all(
    omics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    drugs_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align all tables to shared cell lines and drugs."""
    shared_cells = set(omics_df["cell_line"]) & set(labels_df["cell_line"]) & set(metadata_df["cell_line"])
    shared_drugs = set(drugs_df["drug"]) & set(labels_df["drug"])

    omics_df = omics_df[omics_df["cell_line"].isin(shared_cells)].reset_index(drop=True)
    metadata_df = metadata_df[metadata_df["cell_line"].isin(shared_cells)].reset_index(drop=True)
    labels_df = labels_df[labels_df["cell_line"].isin(shared_cells) & labels_df["drug"].isin(shared_drugs)]
    labels_df = labels_df.reset_index(drop=True)
    drugs_df = drugs_df[drugs_df["drug"].isin(shared_drugs)].reset_index(drop=True)

    missing_cells = set(labels_df["cell_line"]) - set(omics_df["cell_line"])
    missing_drugs = set(labels_df["drug"]) - set(drugs_df["drug"])
    if missing_cells:
        raise ValueError(f"Labels reference cell lines missing from omics: {missing_cells}")
    if missing_drugs:
        raise ValueError(f"Labels reference drugs missing from drug features: {missing_drugs}")

    logger.info(
        "Aligned shapes - omics: %s, labels: %s, drugs: %s, metadata: %s",
        omics_df.shape,
        labels_df.shape,
        drugs_df.shape,
        metadata_df.shape,
    )
    return omics_df, labels_df, drugs_df, metadata_df


def preprocess_gdsc2(raw_dir: str, processed_dir: str, n_genes: int = 2000, fingerprint_bits: int = 1024) -> None:
    """End-to-end preprocessing for GDSC2 raw files."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    ensure_dir(processed_path)

    gdsc2_path = raw_path / "GDSC2_fitted_dose_response_27Oct23.xlsx"
    cell_lines_path = raw_path / "Cell_Lines_Details.xlsx"
    compounds_path = raw_path / "screened_compounds_rel_8.5.csv"
    expression_path = raw_path / "TableS1A.xlsx"

    logger.info("Loading raw GDSC2 files from %s", raw_path)
    gdsc2_df = load_gdsc2_fitted(str(gdsc2_path))
    cell_lines_df = load_cell_line_details(str(cell_lines_path))
    compounds_df = load_screened_compounds(str(compounds_path))
    expression_df = load_expression_table(str(expression_path))

    metadata_df = build_metadata(cell_lines_df)
    drugs_df = build_drug_table(compounds_df)
    labels_df = build_labels(gdsc2_df, metadata_df, drugs_df)
    omics_df = build_expression(expression_df, metadata_df, n_genes=n_genes)

    omics_df, labels_df, drugs_df, metadata_df = align_all(omics_df, labels_df, drugs_df, metadata_df)

    logger.info("Featurizing drugs into Morgan fingerprints (%d bits)", fingerprint_bits)
    drugs_df_with_smiles = drugs_df.copy()
    if "smiles" not in drugs_df_with_smiles.columns:
        drugs_df_with_smiles["smiles"] = np.nan
    drug_fp_df = featurize_drug_table(drugs_df_with_smiles[["drug", "smiles"]], n_bits=fingerprint_bits)

    save_parquet(omics_df, processed_path / "omics.parquet")
    save_parquet(drug_fp_df, processed_path / "drug_fingerprints.parquet")
    save_parquet(labels_df, processed_path / "labels.parquet")
    save_parquet(metadata_df, processed_path / "metadata.parquet")
    logger.info("Saved processed tables to %s", processed_path)
