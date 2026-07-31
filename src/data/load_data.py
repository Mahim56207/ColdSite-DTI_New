"""
Track A (124AD0008) -- load DAVIS/KIBA directly from the original DeepDTA
data files. This bypasses PyTDC entirely, which turned out to have a heavy,
fragile dependency chain on Windows. Uses the exact loading logic documented
in DeepDTA's own README (github.com/hkmztrk/DeepDTA).
"""
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

DEEPDTA_DATA_PATH = "src/data/baselines/deepdta/data"


def load_deepdta_dataset(name: str, deepdta_path: str = DEEPDTA_DATA_PATH) -> pd.DataFrame:
    assert name in ("davis", "kiba"), "name must be 'davis' or 'kiba' (lowercase)"
    fpath = f"{deepdta_path}/{name}/"

    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    Y = pickle.load(open(fpath + "Y", "rb"), encoding="latin1")

    if name == "davis":
        Y = -np.log10(Y / 1e9)

    drug_ids = list(ligands.keys())
    drug_smiles = list(ligands.values())
    target_ids = list(proteins.keys())
    target_seqs = list(proteins.values())

    row_inds, col_inds = np.where(~np.isnan(Y))

    df = pd.DataFrame({
        "Drug_ID": [drug_ids[i] for i in row_inds],
        "Drug": [drug_smiles[i] for i in row_inds],
        "Target_ID": [target_ids[j] for j in col_inds],
        "Target": [target_seqs[j] for j in col_inds],
        "Y": Y[row_inds, col_inds],
    })
    return df


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"{name}: {len(df)} measured pairs, "
          f"{df['Drug_ID'].nunique()} unique drugs, "
          f"{df['Target_ID'].nunique()} unique targets, "
          f"Y range [{df['Y'].min():.3f}, {df['Y'].max():.3f}]")


if __name__ == "__main__":
    for name in ("davis", "kiba"):
        df = load_deepdta_dataset(name)
        summarize(df, name)