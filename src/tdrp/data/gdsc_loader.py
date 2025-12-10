from __future__ import annotations

from typing import Optional
import logging
import pandas as pd
import numpy as np

from tdrp.featurizers.drugs import featurize_drug_table
from tdrp.utils.io import save_parquet, ensure_dir


logger = logging.getLogger(__name__)


def load_expression_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_top_variance_genes(df: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    """Select genes with highest variance across samples."""
    gene_cols = [c for c in df.columns if c != "cell_line"]
    variances = df[gene_cols].var().sort_values(ascending=False)
    selected = variances.head(n_genes).index.tolist()
    cols = ["cell_line"] + selected
    return df[cols]


def zscore_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score gene expression per gene column."""
    result = df.copy()
    gene_cols = [c for c in result.columns if c != "cell_line"]
    result[gene_cols] = (result[gene_cols] - result[gene_cols].mean()) / result[gene_cols].std(ddof=0)
    return result


def load_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_drug_smiles(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_metadata(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    return pd.read_csv(path)


def _align_tables(
    omics_df: pd.DataFrame,
    drug_fp_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    omics_available = set(omics_df["cell_line"])
    drug_available = set(drug_fp_df["drug"])
    labels_mask = labels_df["cell_line"].isin(omics_available) & labels_df["drug"].isin(drug_available)
    filtered_labels = labels_df[labels_mask].reset_index(drop=True)
    missing = len(labels_df) - len(filtered_labels)
    if missing > 0:
        logger.warning("Dropped %d label rows due to missing omics or drug features.", missing)
    used_cell_lines = filtered_labels["cell_line"].unique()
    used_drugs = filtered_labels["drug"].unique()
    omics_df = omics_df[omics_df["cell_line"].isin(used_cell_lines)].reset_index(drop=True)
    drug_fp_df = drug_fp_df[drug_fp_df["drug"].isin(used_drugs)].reset_index(drop=True)
    if metadata_df is not None:
        metadata_df = metadata_df[metadata_df["cell_line"].isin(used_cell_lines)].reset_index(drop=True)
    return omics_df, drug_fp_df, filtered_labels, metadata_df


def preprocess_gdsc(
    expression_path: str,
    labels_path: str,
    drug_smiles_path: str,
    metadata_path: Optional[str],
    outdir: str,
    n_genes: int,
    fingerprint_bits: int,
) -> None:
    """
    Preprocess GDSC-style data and save parquet files.
    """
    ensure_dir(outdir)
    logger.info("Loading expression from %s", expression_path)
    expr = load_expression_table(expression_path)
    expr = select_top_variance_genes(expr, n_genes=n_genes)
    expr = zscore_genes(expr)

    logger.info("Loading labels from %s", labels_path)
    labels = load_labels(labels_path)
    logger.info("Loading drug SMILES from %s", drug_smiles_path)
    drug_smiles = load_drug_smiles(drug_smiles_path)
    logger.info("Featurizing drugs with %d bits", fingerprint_bits)
    drug_fp = featurize_drug_table(drug_smiles, n_bits=fingerprint_bits)

    metadata_df = load_metadata(metadata_path)
    omics_df, drug_fp_df, labels_df, metadata_df = _align_tables(expr, drug_fp, labels, metadata_df)

    save_parquet(omics_df, f"{outdir}/omics.parquet")
    save_parquet(drug_fp_df, f"{outdir}/drug_fingerprints.parquet")
    save_parquet(labels_df, f"{outdir}/labels.parquet")
    if metadata_df is not None:
        save_parquet(metadata_df, f"{outdir}/metadata.parquet")
    logger.info("Preprocessing complete. Saved to %s", outdir)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # Example sanity check can be added here.
