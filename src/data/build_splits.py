"""
Track A (124AD0008) — build the four difficulty splits.

This is the single most important file in the whole repo: it creates the
warm / cold-drug / cold-target / cold-pair splits that ColdSite-DTI's entire
headline result depends on. See docs/01_GUIDE_124AD0008.md Step 2.
"""
import os
import pandas as pd
from src.data.load_data import load_dataset

SPLIT_DIR = "data/splits"


def build_all_splits(dataset_name: str):
    """
    Builds and saves all four splits for the given dataset.
    Returns a dict of {split_name: {'train':df, 'valid':df, 'test':df}}.
    """
    data = load_dataset(dataset_name)

    splits = {
        "random": data.get_split(),
        "cold_drug": data.get_split(method="cold_split", column_name="Drug"),
        "cold_target": data.get_split(method="cold_split", column_name="Target"),
        "cold_pair": data.get_split(method="cold_split", column_name=["Drug", "Target"]),
    }

    out_dir = os.path.join(SPLIT_DIR, dataset_name.lower())
    os.makedirs(out_dir, exist_ok=True)

    for split_name, parts in splits.items():
        for part_name, df in parts.items():
            path = os.path.join(out_dir, f"{split_name}_{part_name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {path}  ({len(df)} rows)")

    return splits


def check_no_leakage(splits: dict) -> None:
    """
    Sanity check that MUST pass before anyone trains on these splits.
    Verifies that for cold splits, no drug/target ID appears in both
    train and test.
    """
    for split_name, parts in splits.items():
        if split_name == "random":
            continue  # random split is allowed to share IDs across sets
        train_drugs = set(parts["train"]["Drug_ID"])
        test_drugs = set(parts["test"]["Drug_ID"])
        train_targets = set(parts["train"]["Target_ID"])
        test_targets = set(parts["test"]["Target_ID"])

        if split_name in ("cold_drug", "cold_pair"):
            overlap = train_drugs & test_drugs
            assert not overlap, f"LEAKAGE in {split_name}: {len(overlap)} drugs in both train and test!"
        if split_name in ("cold_target", "cold_pair"):
            overlap = train_targets & test_targets
            assert not overlap, f"LEAKAGE in {split_name}: {len(overlap)} targets in both train and test!"

        print(f"{split_name}: no leakage detected ✓")


if __name__ == "__main__":
    for dataset_name in ("DAVIS", "KIBA"):
        print(f"\n=== Building splits for {dataset_name} ===")
        splits = build_all_splits(dataset_name)
        check_no_leakage(splits)
