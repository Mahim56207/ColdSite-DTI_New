"""
Track A (124AD0008) -- build the four difficulty splits, without PyTDC.
"""
import os

import numpy as np
import pandas as pd

from src.data.load_data import load_deepdta_dataset

SPLIT_DIR = "data/splits"


def random_split(df: pd.DataFrame, frac_test: float = 0.2, seed: int = 42) -> dict:
    test_df = df.sample(frac=frac_test, random_state=seed)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    return {"train": train_df, "test": test_df.reset_index(drop=True)}


def cold_split(df: pd.DataFrame, cold_column: str, frac_test: float = 0.2, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    unique_ids = df[cold_column].unique()
    n_test = max(1, int(len(unique_ids) * frac_test))
    test_ids = set(rng.choice(unique_ids, size=n_test, replace=False))

    test_df = df[df[cold_column].isin(test_ids)].reset_index(drop=True)
    train_df = df[~df[cold_column].isin(test_ids)].reset_index(drop=True)
    return {"train": train_df, "test": test_df}


def cold_pair_split(df: pd.DataFrame, frac_test: float = 0.2, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    drug_ids = df["Drug_ID"].unique()
    target_ids = df["Target_ID"].unique()

    n_test_drugs = max(1, int(len(drug_ids) * frac_test))
    n_test_targets = max(1, int(len(target_ids) * frac_test))
    test_drugs = set(rng.choice(drug_ids, size=n_test_drugs, replace=False))
    test_targets = set(rng.choice(target_ids, size=n_test_targets, replace=False))

    is_test_drug = df["Drug_ID"].isin(test_drugs)
    is_test_target = df["Target_ID"].isin(test_targets)

    test_df = df[is_test_drug & is_test_target].reset_index(drop=True)
    train_df = df[~is_test_drug & ~is_test_target].reset_index(drop=True)
    return {"train": train_df, "test": test_df}


def build_all_splits(dataset_name: str) -> dict:
    df = load_deepdta_dataset(dataset_name)
    splits = {
        "random": random_split(df),
        "cold_drug": cold_split(df, "Drug_ID"),
        "cold_target": cold_split(df, "Target_ID"),
        "cold_pair": cold_pair_split(df),
    }
    out_dir = os.path.join(SPLIT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for split_name, parts in splits.items():
        for part_name, part_df in parts.items():
            path = os.path.join(out_dir, f"{split_name}_{part_name}.csv")
            part_df.to_csv(path, index=False)
            print(f"Saved {path}  ({len(part_df)} rows)")
    return splits


def check_no_leakage(splits: dict) -> None:
    for split_name, parts in splits.items():
        if split_name == "random":
            continue
        train_drugs, test_drugs = set(parts["train"]["Drug_ID"]), set(parts["test"]["Drug_ID"])
        train_targets, test_targets = set(parts["train"]["Target_ID"]), set(parts["test"]["Target_ID"])
        if split_name in ("cold_drug", "cold_pair"):
            assert not (train_drugs & test_drugs), f"LEAKAGE in {split_name}!"
        if split_name in ("cold_target", "cold_pair"):
            assert not (train_targets & test_targets), f"LEAKAGE in {split_name}!"
        print(f"{split_name}: no leakage detected -- OK")


if __name__ == "__main__":
    for dataset_name in ("davis", "kiba"):
        print(f"\n=== Building splits for {dataset_name} ===")
        splits = build_all_splits(dataset_name)
        check_no_leakage(splits)
        for split_name, parts in splits.items():
            print(f"  {split_name}: train={len(parts['train'])}  test={len(parts['test'])}")