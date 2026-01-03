from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from tdrp.utils.io import ensure_dir

logger = logging.getLogger(__name__)


PUBCHEM_PROPERTIES = (
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "InChIKey",
)


def _import_drugs():
    try:
        from drugs import Drug
        from drugs.data_sources import pubchem_properties
    except Exception as exc:
        raise ImportError(
            "Optional dependency 'drugs' is required. Install with: pip install drugs==0.1.1"
        ) from exc
    return Drug, pubchem_properties


def _load_smiles_report(processed_dir: Path) -> pd.DataFrame:
    report_path = processed_dir / "smiles_match_report.csv"
    if report_path.exists():
        report = pd.read_csv(report_path)
        return report[["drug", "drug_name", "synonyms", "smiles"]]
    return pd.DataFrame(columns=["drug", "drug_name", "synonyms", "smiles"])


def _load_raw_drugs(raw_dir: Path) -> pd.DataFrame:
    raw_path = raw_dir / "screened_compounds_rel_8.5.csv"
    if not raw_path.exists():
        return pd.DataFrame(columns=["drug", "drug_name", "synonyms", "smiles"])
    raw = pd.read_csv(raw_path)
    cols = {c.lower(): c for c in raw.columns}
    drug_id_col = cols.get("drug_id", "DRUG_ID")
    name_col = cols.get("drug_name", "DRUG_NAME")
    synonyms_col = cols.get("synonyms", "SYNONYMS")
    return pd.DataFrame(
        {
            "drug": raw[drug_id_col].astype(str),
            "drug_name": raw.get(name_col),
            "synonyms": raw.get(synonyms_col),
            "smiles": None,
        }
    )


def _split_synonyms(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _smiles_to_inchikey(smiles: Any) -> Optional[str]:
    if smiles is None or pd.isna(smiles):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi
    except Exception:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _summarize_mechanism(mechanism: dict) -> Optional[str]:
    if not mechanism:
        return None
    parts = []
    for key in ("mechanism_of_action", "action_type", "target_chembl_id", "target_name"):
        value = mechanism.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts) if parts else None


def _extract_target_detail(drug_obj, mechanisms: Iterable[dict]) -> tuple[Optional[str], Optional[str]]:
    for mechanism in mechanisms:
        target_id = mechanism.get("target_chembl_id")
        if not target_id:
            continue
        try:
            detail = drug_obj.fetch_target_details(target_id)
        except Exception:
            continue
        for component in detail.get("target_components", []) or []:
            accession = component.get("accession")
            gene_symbol = None
            for synonym in component.get("target_component_synonyms", []) or []:
                if synonym.get("syn_type") == "GENE_SYMBOL" and synonym.get("component_synonym"):
                    gene_symbol = synonym["component_synonym"]
                    break
            return accession, gene_symbol
    return None, None


def enrich_drugs(
    processed_dir: str | Path = "data/processed",
    output_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    max_drugs: Optional[int] = None,
    sleep_sec: float = 0.0,
) -> Path:
    Drug, pubchem_properties = _import_drugs()
    processed_dir = Path(processed_dir)
    output_path = Path(output_path) if output_path else processed_dir / "drug_metadata_enriched.parquet"

    fingerprints = pd.read_parquet(processed_dir / "drug_fingerprints.parquet")
    drug_ids = fingerprints[["drug"]].drop_duplicates().copy()
    drug_ids["drug"] = drug_ids["drug"].astype(str)

    report = _load_smiles_report(processed_dir)
    if report.empty:
        raw_dir = processed_dir.parent / "raw"
        report = _load_raw_drugs(raw_dir)

    report["drug"] = report["drug"].astype(str)
    base = drug_ids.merge(report, on="drug", how="left")

    if max_drugs:
        base = base.head(max_drugs)

    ensure_dir(output_path.parent)
    if report_dir:
        ensure_dir(report_dir)

    cache: dict[str, dict[str, Any]] = {}
    enriched_rows = []

    total = len(base)
    for idx, (_, row) in enumerate(base.iterrows(), start=1):
        drug_id = row.get("drug")
        drug_name = row.get("drug_name")
        synonyms = row.get("synonyms")
        smiles = row.get("smiles")
        inchikey = _smiles_to_inchikey(smiles)

        cached = cache.get(inchikey) if inchikey else None
        if cached is not None:
            enriched = cached.copy()
        else:
            enriched = {
                "pubchem_cid": None,
                "chembl_id": None,
                "inchikey": inchikey,
                "pubchem_logp": None,
                "pubchem_tpsa": None,
                "pubchem_hbd": None,
                "pubchem_hba": None,
                "pubchem_rotb": None,
                "pubchem_mw": None,
                "chembl_primary_target_accession": None,
                "chembl_primary_target_gene": None,
                "chembl_mechanism_summary": None,
            }
            drug_obj = None
            try:
                drug_obj = Drug(_inchikey=inchikey, synonyms=_split_synonyms(synonyms))
                ids = drug_obj.map_ids()
                enriched["pubchem_cid"] = ids.get("pubchem_cid")
                enriched["chembl_id"] = ids.get("chembl_id")
                enriched["inchikey"] = ids.get("inchikey")
            except Exception as exc:
                logger.warning("ID mapping failed for drug %s: %s", drug_id, exc)

            if enriched["pubchem_cid"]:
                try:
                    props = pubchem_properties(int(enriched["pubchem_cid"]), properties=PUBCHEM_PROPERTIES)
                    enriched["pubchem_logp"] = props.get("XLogP")
                    enriched["pubchem_tpsa"] = props.get("TPSA")
                    enriched["pubchem_hbd"] = props.get("HBondDonorCount")
                    enriched["pubchem_hba"] = props.get("HBondAcceptorCount")
                    enriched["pubchem_rotb"] = props.get("RotatableBondCount")
                    enriched["pubchem_mw"] = props.get("MolecularWeight")
                    if not enriched["inchikey"] and props.get("InChIKey"):
                        enriched["inchikey"] = props.get("InChIKey")
                except Exception as exc:
                    logger.warning("PubChem properties failed for drug %s: %s", drug_id, exc)

            if enriched["chembl_id"] and drug_obj is not None:
                try:
                    mechanisms = drug_obj.fetch_chembl_mechanisms(limit=25)
                    enriched["chembl_mechanism_summary"] = _summarize_mechanism(mechanisms[0]) if mechanisms else None
                    acc, gene = _extract_target_detail(drug_obj, mechanisms)
                    enriched["chembl_primary_target_accession"] = acc
                    enriched["chembl_primary_target_gene"] = gene
                except Exception as exc:
                    logger.warning("ChEMBL fetch failed for drug %s: %s", drug_id, exc)

            if report_dir and drug_obj is not None:
                try:
                    safe_name = str(drug_name) if drug_name else "unknown"
                    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_name)
                    report_path = Path(report_dir) / f"drug_{drug_id}_{safe_name}.md"
                    drug_obj.write_drug_markdown(output_path=report_path)
                except Exception as exc:
                    logger.warning("Markdown report failed for drug %s: %s", drug_id, exc)

            if inchikey:
                cache[inchikey] = enriched.copy()

        enriched_rows.append(
            {
                "drug": drug_id,
                "drug_name": drug_name,
                "synonyms": synonyms,
                "smiles": smiles,
                **enriched,
            }
        )

        if idx == 1 or idx % 10 == 0 or idx == total:
            logger.info("Enriched %s/%s drugs (latest=%s)", idx, total, drug_id)

        if sleep_sec:
            time.sleep(sleep_sec)

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_parquet(output_path, index=False)
    logger.info("Saved drug metadata to %s", output_path)
    return output_path
