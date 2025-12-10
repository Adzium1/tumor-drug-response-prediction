from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError as exc:
    raise ImportError(
        "RDKit is required for drug featurization. Install via conda: conda install -c conda-forge rdkit"
    ) from exc


logger = logging.getLogger(__name__)


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """Convert SMILES string to Morgan fingerprint bit vector."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        logger.warning("Missing SMILES string encountered; returning zeros.")
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        logger.warning("Invalid SMILES string encountered: %s", smiles)
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)  # type: ignore[arg-type]
    return arr.astype(np.float32)


def featurize_drug_table(
    drug_df: pd.DataFrame,
    n_bits: int = 1024,
) -> pd.DataFrame:
    """
    Input: DataFrame with columns ['drug', 'smiles'].
    Output: DataFrame with ['drug', fp_0, ..., fp_{n_bits-1}]
    """
    fps = []
    for _, row in drug_df.iterrows():
        fp = smiles_to_morgan_fp(row.get("smiles"), n_bits=n_bits)
        fps.append(fp)
    fp_array = np.stack(fps)
    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    result = pd.DataFrame(fp_array, columns=fp_cols)
    result.insert(0, "drug", drug_df["drug"].values)
    return result


def smiles_to_molecular_graph(smiles: str):
    """Stub for graph featurization."""
    raise NotImplementedError("Graph featurization not implemented yet.")


# if __name__ == "__main__":
#     example = featurize_drug_table(pd.DataFrame({"drug": ["x"], "smiles": ["CCO"]}), n_bits=16)
#     print(example.head())
