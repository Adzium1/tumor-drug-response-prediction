from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from tdrp.featurizers.drugs import featurize_drug_table
from tdrp.utils.io import ensure_dir, save_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, strip, replace spaces/newlines with underscores."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\n", "_").replace(" ", "_") for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> Optional[str]:
    normalized = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    logger.warning("Expected column %s in %s, got: %s", list(candidates), context, list(df.columns))
    return None


def _require_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> str:
    col = _find_column(df, candidates, context)
    if col is None:
        raise ValueError(f"Expected column {list(candidates)} in {context}, got: {list(df.columns)}")
    return col


def _normalize_id(value) -> Optional[str]:
    if pd.isna(value):
        return None
    try:
        return str(int(float(str(value))))
    except Exception:
        return str(value).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_cell_line_metadata(path: str) -> pd.DataFrame:
    """Load and tidy cell line metadata."""
    df = normalize_columns(pd.read_excel(path, engine="openpyxl"))
    cosmic_col = _require_column(df, ["cosmic_identifier", "cosmic_id", "cosmic"], "cell line metadata")
    name_col = _require_column(df, ["sample_name", "cell_line_name", "line", "cell_line"], "cell line metadata")
    tissue_col = _find_column(df, ["gdsc_tissue_descriptor_1", "tissue"], "cell line metadata")
    site_col = _find_column(df, ["gdsc_tissue_descriptor_2", "site"], "cell line metadata")
    cancer_col = _find_column(df, ["cancer_type_(matching_tcga_label)", "cancer_type"], "cell line metadata")

    meta = pd.DataFrame(
        {
            "cosmic_id": pd.to_numeric(df[cosmic_col], errors="coerce").astype("Int64"),
            "cell_line": df[name_col].astype(str),
        }
    )
    if tissue_col:
        meta["tissue"] = df[tissue_col].astype(str)
    if site_col:
        meta["site"] = df[site_col].astype(str)
    if cancer_col:
        meta["cancer_type"] = df[cancer_col].astype(str)

    meta = meta.dropna(subset=["cosmic_id", "cell_line"]).drop_duplicates(subset=["cosmic_id"])
    meta = meta.reset_index(drop=True)
    return meta


def load_gdsc2_drugs(path: str) -> pd.DataFrame:
    """Load drug table from screened compounds."""
    df = normalize_columns(pd.read_csv(path))
    drug_id_col = _require_column(df, ["drug_id"], "screened compounds")
    name_col = _find_column(df, ["drug_name", "drug"], "screened compounds")
    target_col = _find_column(df, ["putative_target", "target", "target_pathway"], "screened compounds")
    smiles_col = _find_column(df, ["smiles", "canonical_smiles"], "screened compounds")

    drugs = pd.DataFrame({"drug": df[drug_id_col].astype(str)})
    if name_col:
        drugs["drug_name"] = df[name_col]
    if target_col:
        drugs["putative_target"] = df[target_col]
    drugs["smiles"] = df[smiles_col] if smiles_col else np.nan

    drugs = drugs.drop_duplicates(subset=["drug"]).reset_index(drop=True)
    return drugs


def load_gdsc2_dose_response(path: str, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load GDSC2 fitted dose-response and return labels table and mapping."""
    df = normalize_columns(pd.read_excel(path, engine="openpyxl"))
    cosmic_col = _require_column(df, ["cosmic_id", "cosmic", "cosmic_identifier"], "dose-response")
    drug_col = _require_column(df, ["drug_id"], "dose-response")
    sanger_col = _require_column(df, ["sanger_model_id"], "dose-response")
    cell_line_name_col = _find_column(df, ["cell_line_name"], "dose-response")
    ln_ic50_col = _find_column(df, ["ln_ic50", "ln_ic50_um", "ln_ic50_(um)", "ln_ic50_umol", "ln_ic50_(umol)"], "dose-response")
    ic50_col = _find_column(df, ["ic50", "ic50_um", "ic50_(um)", "ic50_umol"], "dose-response")
    if ln_ic50_col is None and ic50_col is None:
        raise ValueError("Dose-response file missing ln_ic50/ic50 column.")

    labels = pd.DataFrame()
    labels["drug"] = df[drug_col].astype(str)
    labels["cell_line"] = df[sanger_col].astype(str)

    if ln_ic50_col:
        labels["ln_ic50"] = pd.to_numeric(df[ln_ic50_col], errors="coerce")
    else:
        ic50 = pd.to_numeric(df[ic50_col], errors="coerce")
        ic50 = ic50.replace({0: np.nan})
        labels["ln_ic50"] = np.log(ic50)

    before = len(labels)
    labels = labels.dropna(subset=["cell_line", "drug", "ln_ic50"])
    labels = labels[np.isfinite(labels["ln_ic50"])]
    dropped = before - len(labels)
    if dropped:
        logger.warning("Dropped %d dose-response rows due to missing mappings/values.", dropped)

    labels = labels.reset_index(drop=True)
    labels = labels[["cell_line", "drug", "ln_ic50"]]

    mapping_cols = {
        "cosmic_id": df[cosmic_col].apply(_normalize_id),
        "cell_line": df[sanger_col].astype(str),
    }
    if cell_line_name_col:
        mapping_cols["cell_line_name"] = df[cell_line_name_col]
    map_df = pd.DataFrame(mapping_cols).dropna(subset=["cosmic_id", "cell_line"]).drop_duplicates()
    map_df["cosmic_id"] = pd.to_numeric(map_df["cosmic_id"], errors="coerce").astype("Int64")

    return labels, map_df


def _select_top_genes(df: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    variances = df.var(axis=0, ddof=0)
    top = variances.sort_values(ascending=False).head(min(n_genes, len(variances))).index
    return df[top]


def load_rnaseq_expression(
    root_dir: str,
    metadata: pd.DataFrame,
    allowed_cell_lines: Optional[set[str]] = None,
    n_genes: int = 2000,
) -> pd.DataFrame:
    """Load RNA-seq expression, map COSMIC->cell_line, keep top-variance genes, z-score."""
    root = Path(root_dir)
    # Prefer FPKM file; fall back to read counts.
    expr_files = sorted(
        list(root.glob("rnaseq_fpkm_20191101.*"))
        + list(root.glob("rnaseq_read_count_20191101.*"))
        + list(root.glob("rnaseq_*20191101*.txt"))
        + list(root.glob("rnaseq_*20191101*.csv"))
    )
    if not expr_files:
        raise FileNotFoundError(f"No expression file (.txt or .csv) found under {root}")
    # Prefer fpkm if present
    fpkm_files = [p for p in expr_files if "fpkm" in p.name.lower()]
    expr_file = fpkm_files[0] if fpkm_files else expr_files[0]
    sep = "\t" if expr_file.suffix == ".txt" else ","
    df = pd.read_csv(expr_file, sep=sep, low_memory=False)
    logger.info("Loaded expression file %s with shape %s", expr_file.name, df.shape)

    # File layout: first two columns are gene_id, gene_symbol; first three rows are metadata
    data = df.iloc[3:].copy()
    data.columns = df.columns
    gene_id_col, gene_symbol_col = data.columns[:2]
    data.index = data[gene_symbol_col].fillna(data[gene_id_col])
    data = data.drop(columns=[gene_id_col, gene_symbol_col])
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.transpose()
    data.index.name = "cell_line"
    data.reset_index(inplace=True)

    expr = data.rename(columns={"index": "cell_line"})
    if allowed_cell_lines is not None:
        expr = expr[expr["cell_line"].isin(allowed_cell_lines)]

    expr = expr.set_index("cell_line")
    expr = expr.dropna(axis=1, how="all")
    expr = _select_top_genes(expr, n_genes=n_genes)
    expr = (expr - expr.mean()) / expr.std(ddof=0)
    expr = expr.astype(np.float32)
    expr = expr.reset_index()
    return expr


# ---------------------------------------------------------------------------
# Alignment and preprocessing entrypoint
# ---------------------------------------------------------------------------
def align_all(
    omics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    drugs_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align tables on shared cell lines/drugs and validate keys."""
    shared_cells = set(omics_df["cell_line"]) & set(labels_df["cell_line"]) & set(metadata_df["cell_line"])
    shared_drugs = set(drugs_df["drug"]) & set(labels_df["drug"])

    omics_df = omics_df[omics_df["cell_line"].isin(shared_cells)].reset_index(drop=True)
    metadata_df = metadata_df[metadata_df["cell_line"].isin(shared_cells)].reset_index(drop=True)
    labels_df = labels_df[labels_df["cell_line"].isin(shared_cells) & labels_df["drug"].isin(shared_drugs)].reset_index(drop=True)
    drugs_df = drugs_df[drugs_df["drug"].isin(shared_drugs)].reset_index(drop=True)

    missing_cells = set(labels_df["cell_line"]) - set(omics_df["cell_line"])
    missing_drugs = set(labels_df["drug"]) - set(drugs_df["drug"])
    if missing_cells:
        raise ValueError(f"Labels reference cell lines missing from omics: {missing_cells}")
    if missing_drugs:
        raise ValueError(f"Labels reference drugs missing from drug table: {missing_drugs}")

    logger.info(
        "Aligned shapes - omics: %s, labels: %s, drugs: %s, metadata: %s",
        omics_df.shape,
        labels_df.shape,
        drugs_df.shape,
        metadata_df.shape,
    )
    return omics_df, labels_df, drugs_df, metadata_df


def preprocess_gdsc2(raw_dir: str, processed_dir: str, n_genes: int = 2000, fingerprint_bits: int = 1024) -> None:
    """End-to-end preprocessing of GDSC2 raw files into parquet tables."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    ensure_dir(processed_path)

    meta_path = raw_path / "Cell_Lines_Details.xlsx"
    dose_path = raw_path / "GDSC2_fitted_dose_response_27Oct23.xlsx"
    drugs_path = raw_path / "screened_compounds_rel_8.5.csv"
    expr_dir = raw_path

    logger.info("Loading cell line metadata...")
    metadata_df = load_cell_line_metadata(str(meta_path))
    logger.info("Loading dose-response...")
    labels_df, map_df = load_gdsc2_dose_response(str(dose_path), metadata_df)
    logger.info("Building metadata with SANGER_MODEL_ID...")
    metadata_df = metadata_df.merge(map_df, on="cosmic_id", how="inner", suffixes=("_meta", "_map"))
    if "cell_line_map" in metadata_df.columns:
        metadata_df["cell_line"] = metadata_df["cell_line_map"]
    elif "cell_line" in metadata_df.columns:
        metadata_df["cell_line"] = metadata_df["cell_line"]
    if "cell_line_meta" in metadata_df.columns:
        metadata_df["cell_line_original"] = metadata_df["cell_line_meta"]
    elif "cell_line_name" in metadata_df.columns:
        metadata_df["cell_line_original"] = metadata_df["cell_line_name"]
    metadata_df = metadata_df.drop(columns=[c for c in ["cell_line_map", "cell_line_meta"] if c in metadata_df.columns])
    metadata_df = metadata_df.drop_duplicates(subset=["cell_line"])

    logger.info("Loading drug table...")
    drugs_df = load_gdsc2_drugs(str(drugs_path))
    logger.info("Loading RNA-seq expression...")
    omics_df = load_rnaseq_expression(str(expr_dir), metadata_df, allowed_cell_lines=set(labels_df["cell_line"]), n_genes=n_genes)

    omics_df, labels_df, drugs_df, metadata_df = align_all(omics_df, labels_df, drugs_df, metadata_df)

    logger.info("Featurizing drugs into %d-bit Morgan fingerprints", fingerprint_bits)
    drug_fp_df = featurize_drug_table(drugs_df[["drug", "smiles"]], n_bits=fingerprint_bits)

    save_parquet(omics_df, processed_path / "omics.parquet")
    save_parquet(drug_fp_df, processed_path / "drug_fingerprints.parquet")
    save_parquet(labels_df, processed_path / "labels.parquet")
    save_parquet(metadata_df, processed_path / "metadata.parquet")

    logger.info(
        "Saved processed tables to %s (omics %s, drug_fingerprints %s, labels %s, metadata %s)",
        processed_path,
        omics_df.shape,
        drug_fp_df.shape,
        labels_df.shape,
        metadata_df.shape,
    )
