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


def _normalize_cell_line_name(value: str) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _strip_cell_line_descriptor(value: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if "," in text:
        text = text.split(",", 1)[0]
    return text.strip()


def _normalize_drug_name(value: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    for ch in ["-", "/", "\\", "(", ")", "[", "]", "{", "}", "'", '"']:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _split_synonyms(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


_SALT_TOKENS = {
    "HCL",
    "HBR",
    "H2SO4",
    "H2SO3",
    "HNO3",
    "HYDROCHLORIDE",
    "HYDROBROMIDE",
    "SULFATE",
    "SULPHATE",
    "PHOSPHATE",
    "NITRATE",
    "ACETATE",
    "MALEATE",
    "FUMARATE",
    "TARTRATE",
    "CITRATE",
    "OXALATE",
    "MESYLATE",
    "TOSYLATE",
    "BESYLATE",
    "TRIFLUOROACETATE",
    "TFA",
    "SODIUM",
    "POTASSIUM",
    "CALCIUM",
    "MAGNESIUM",
    "CHLORIDE",
    "BROMIDE",
    "IODIDE",
}


def _normalize_drug_key(value: str, drop_salts: bool = False) -> str:
    base = _normalize_drug_name(value)
    tokens = base.split()
    if drop_salts:
        tokens = [t for t in tokens if t not in _SALT_TOKENS]
    return " ".join(tokens)


def _is_missing_smiles(value) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _build_name_map(map_df: pd.DataFrame) -> dict[str, str]:
    """Map normalized cell line names to SANGER_MODEL_ID values."""
    if "cell_line_name" not in map_df.columns:
        return {}
    subset = map_df[["cell_line_name", "cell_line"]].dropna()
    name_map = {}
    for _, row in subset.iterrows():
        key = _normalize_cell_line_name(row["cell_line_name"])
        if key:
            name_map[key] = str(row["cell_line"])
    return name_map


def _maybe_drop_metadata_rows(df: pd.DataFrame, expr_cols: list[str]) -> pd.DataFrame:
    """Drop the first 3 rows if they look like metadata (non-numeric)."""
    if len(df) < 4:
        return df
    head = df[expr_cols].head(3)
    numeric = head.apply(pd.to_numeric, errors="coerce")
    non_numeric_frac = float(numeric.isna().mean().mean())
    if non_numeric_frac > 0.5:
        return df.iloc[3:].copy()
    return df


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
    synonyms_col = _find_column(df, ["synonyms"], "screened compounds")

    drugs = pd.DataFrame({"drug": df[drug_id_col].astype(str)})
    if name_col:
        drugs["drug_name"] = df[name_col]
    if target_col:
        drugs["putative_target"] = df[target_col]
    if synonyms_col:
        drugs["synonyms"] = df[synonyms_col]
    drugs["smiles"] = df[smiles_col] if smiles_col else np.nan

    drugs = drugs.drop_duplicates(subset=["drug"]).reset_index(drop=True)
    return drugs


def load_gdsc_drug_annotation(path: str) -> pd.DataFrame:
    """Load drug annotation with SMILES, keyed by drug name."""
    df = pd.read_csv(path)
    if df.columns.size > 0:
        first_col = str(df.columns[0])
        if first_col == "" or first_col.lower().startswith("unnamed"):
            df = df.rename(columns={df.columns[0]: "drug_name"})
    df = normalize_columns(df)
    if "drug_name" not in df.columns:
        df = df.rename(columns={df.columns[0]: "drug_name"})
    smiles_col = _find_column(
        df,
        [
            "canonical_smilesrdkit",
            "canonicalsmilesrdkit",
            "canonical_smiles",
            "canonicalsmiles",
            "smiles",
        ],
        "drug annotation",
    )
    if smiles_col is None:
        raise ValueError(f"Drug annotation missing SMILES columns: {list(df.columns)}")
    ann = df[["drug_name", smiles_col]].copy()
    ann = ann.rename(columns={smiles_col: "smiles"})
    ann["drug_name"] = ann["drug_name"].astype(str).str.strip()
    ann["smiles"] = ann["smiles"].astype(str).str.strip()
    ann = ann[ann["drug_name"].ne("") & ann["smiles"].ne("") & ann["smiles"].str.lower().ne("nan")]
    ann = ann.drop_duplicates(subset=["drug_name"])
    ann["drug_name_norm"] = ann["drug_name"].apply(_normalize_drug_key)
    ann["drug_name_norm_nosalt"] = ann["drug_name"].apply(lambda x: _normalize_drug_key(x, drop_salts=True))
    return ann


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
    name_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load RNA-seq expression, map cell line names to SANGER IDs, keep top-variance genes, z-score."""
    root = Path(root_dir)
    # Prefer FPKM file; fall back to read counts.
    expr_files = sorted(
        list(root.glob("rnaseq_fpkm_20191101.*"))
        + list(root.glob("rnaseq_read_count_20191101.*"))
        + list(root.glob("rnaseq_*20191101*.txt"))
        + list(root.glob("rnaseq_*20191101*.csv"))
        + list(root.glob("E-MTAB-3983*.tsv"))
        + list(root.glob("E-MTAB-3983*.txt"))
        + list(root.glob("E-MTAB-3983*.csv"))
    )
    if not expr_files:
        raise FileNotFoundError(f"No expression file (.txt or .csv) found under {root}")
    # Prefer fpkm if present
    fpkm_files = [p for p in expr_files if "fpkm" in p.name.lower()]
    emtab_files = [p for p in expr_files if "e-mtab-3983" in p.name.lower()]
    if fpkm_files:
        expr_file = fpkm_files[0]
    elif emtab_files:
        expr_file = emtab_files[0]
    else:
        expr_file = expr_files[0]
    sep = "\t" if expr_file.suffix.lower() in [".txt", ".tsv"] else ","
    df = pd.read_csv(expr_file, sep=sep, low_memory=False, comment="#")
    logger.info("Loaded expression file %s with shape %s", expr_file.name, df.shape)

    # File layout: first two columns are gene_id, gene_symbol; some files include 3 metadata rows.
    gene_id_col = _find_column(df, ["gene_id", "gene id", "ensembl_gene_id"], "expression")
    gene_symbol_col = _find_column(df, ["gene_name", "gene name", "gene_symbol", "gene symbol"], "expression")
    if gene_id_col is None:
        gene_id_col = df.columns[0]
    if gene_symbol_col is None and len(df.columns) > 1:
        gene_symbol_col = df.columns[1]

    expr_cols = [c for c in df.columns if c not in {gene_id_col, gene_symbol_col}]
    data = _maybe_drop_metadata_rows(df, expr_cols)
    gene_ids = data[gene_id_col].astype(str)
    gene_symbols = data[gene_symbol_col].astype(str) if gene_symbol_col else gene_ids
    gene_labels = gene_symbols.fillna(gene_ids)

    values = data[expr_cols].apply(pd.to_numeric, errors="coerce")
    values.index = gene_labels
    expr = values.transpose()
    expr.index = [_strip_cell_line_descriptor(c) for c in expr.index]
    expr.index.name = "cell_line"

    if name_map:
        mapped = pd.Series(expr.index, index=expr.index).map(lambda x: name_map.get(_normalize_cell_line_name(x)))
        n_mapped = int(mapped.notna().sum())
        logger.info("Mapped %d/%d expression samples to SANGER IDs.", n_mapped, len(expr))
        mask = mapped.notna()
        expr = expr.loc[mask].copy()
        expr.index = mapped[mask].astype(str).values
        expr.index.name = "cell_line"

    if allowed_cell_lines is not None:
        expr = expr[expr.index.isin(allowed_cell_lines)]

    if expr.empty:
        raise ValueError("No expression samples remain after mapping/filtering. Check cell line identifiers.")

    expr = expr.dropna(axis=1, how="all")
    if not expr.index.is_unique:
        expr = expr.groupby(expr.index).mean()
    # Drop unnamed columns and collapse duplicate gene symbols by averaging.
    expr = expr.loc[:, ~expr.columns.isna()]
    if expr.columns.has_duplicates:
        expr = expr.groupby(expr.columns, axis=1).mean()
    max_nan_frac = 0.5
    nan_frac = expr.isna().mean()
    drop_cols = nan_frac[nan_frac > max_nan_frac].index
    if len(drop_cols) > 0:
        logger.info("Dropping %d genes with > %.0f%% missingness.", len(drop_cols), 100 * max_nan_frac)
        expr = expr.drop(columns=drop_cols)
    expr = _select_top_genes(expr, n_genes=n_genes)
    mean = expr.mean()
    std = expr.std(ddof=0).replace(0, np.nan)
    expr = (expr - mean) / std
    expr = expr.fillna(0.0)
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
    name_map = _build_name_map(map_df)
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
    if drugs_df["smiles"].isna().any():
        ann_path = raw_path / "GDSC_DrugAnnotation.csv"
        if ann_path.exists():
            logger.info("Merging drug SMILES from %s", ann_path.name)
            ann_df = load_gdsc_drug_annotation(str(ann_path))
            name_to_smiles = {}
            name_to_smiles_nosalt = {}
            ann_token_entries: list[tuple[set[str], str]] = []
            for _, row in ann_df.iterrows():
                norm = row["drug_name_norm"]
                norm_nosalt = row["drug_name_norm_nosalt"]
                if norm and norm not in name_to_smiles:
                    name_to_smiles[norm] = row["smiles"]
                if norm_nosalt and norm_nosalt not in name_to_smiles_nosalt:
                    name_to_smiles_nosalt[norm_nosalt] = row["smiles"]
                tokens = set(norm_nosalt.split()) if isinstance(norm_nosalt, str) else set()
                if tokens:
                    ann_token_entries.append((tokens, row["smiles"]))
            drugs_df["smiles"] = drugs_df["smiles"].astype("object")
            missing_before = int(drugs_df["smiles"].apply(_is_missing_smiles).sum())
            filled_name = 0
            filled_name_nosalt = 0
            filled_syn = 0
            filled_syn_nosalt = 0
            filled_subset = 0
            match_sources = []
            for idx, row in drugs_df.iterrows():
                if not _is_missing_smiles(row["smiles"]):
                    match_sources.append("existing")
                    continue
                name_norm = _normalize_drug_key(row.get("drug_name", ""))
                name_norm_nosalt = _normalize_drug_key(row.get("drug_name", ""), drop_salts=True)
                smiles = None
                source = "missing"
                if name_norm and name_norm in name_to_smiles:
                    smiles = name_to_smiles[name_norm]
                    source = "name"
                    filled_name += 1
                elif name_norm_nosalt and name_norm_nosalt in name_to_smiles_nosalt:
                    smiles = name_to_smiles_nosalt[name_norm_nosalt]
                    source = "name_nosalt"
                    filled_name_nosalt += 1
                else:
                    for syn in _split_synonyms(row.get("synonyms", None)):
                        syn_norm = _normalize_drug_key(syn)
                        if syn_norm and syn_norm in name_to_smiles:
                            smiles = name_to_smiles[syn_norm]
                            source = "synonym"
                            filled_syn += 1
                            break
                        syn_norm_nosalt = _normalize_drug_key(syn, drop_salts=True)
                        if syn_norm_nosalt and syn_norm_nosalt in name_to_smiles_nosalt:
                            smiles = name_to_smiles_nosalt[syn_norm_nosalt]
                            source = "synonym_nosalt"
                            filled_syn_nosalt += 1
                            break
                if smiles is None:
                    tokens = set(name_norm_nosalt.split())
                    if len(tokens) >= 2:
                        matches = {s for t, s in ann_token_entries if tokens.issubset(t)}
                        if len(matches) == 1:
                            smiles = next(iter(matches))
                            source = "token_subset"
                            filled_subset += 1
                if smiles is not None:
                    drugs_df.at[idx, "smiles"] = smiles
                match_sources.append(source)
            drugs_df["smiles_source"] = match_sources
            filled_total = filled_name + filled_syn
            logger.info(
                "Filled SMILES for %d/%d missing drugs (name=%d, name_nosalt=%d, synonyms=%d, synonyms_nosalt=%d, token_subset=%d).",
                filled_name + filled_name_nosalt + filled_syn + filled_syn_nosalt + filled_subset,
                missing_before,
                filled_name,
                filled_name_nosalt,
                filled_syn,
                filled_syn_nosalt,
                filled_subset,
            )
            report_cols = ["drug", "drug_name", "synonyms", "smiles", "smiles_source"]
            report_df = drugs_df[[c for c in report_cols if c in drugs_df.columns]].copy()
            report_path = processed_path / "smiles_match_report.csv"
            report_df.to_csv(report_path, index=False)
            missing_df = report_df[report_df["smiles_source"] == "missing"]
            missing_path = processed_path / "smiles_unmatched.csv"
            missing_df.to_csv(missing_path, index=False)
            logger.info("Saved SMILES match report to %s", report_path)
            logger.info("Saved unmatched drug list to %s", missing_path)
        else:
            logger.warning("No drug annotation file found at %s; SMILES remain missing.", ann_path)
    logger.info("Loading RNA-seq expression...")
    omics_df = load_rnaseq_expression(
        str(expr_dir),
        metadata_df,
        allowed_cell_lines=set(labels_df["cell_line"]),
        n_genes=n_genes,
        name_map=name_map,
    )

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
