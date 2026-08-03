"""
Track A (124AD0008) -- build the four difficulty splits.

Produces a proper train/valid/test split for each level, saved as:
    data/splits/{dataset}/{split_name}/train.csv
    data/splits/{dataset}/{split_name}/valid.csv
    data/splits/{dataset}/{split_name}/test.csv
This is the directory shape src/model/dataset.py's load_split() expects.
"""
import os

import numpy as np
import pandas as pd

from src.data.load_data import load_deepdta_dataset

SPLIT_DIR = "data/splits"


def random_split(df: pd.DataFrame, frac_valid=0.1, frac_test=0.2, seed=42):
    test_df = df.sample(frac=frac_test, random_state=seed)
    remaining = df.drop(test_df.index)
    valid_df = remaining.sample(frac=frac_valid / (1 - frac_test), random_state=seed)
    train_df = remaining.drop(valid_df.index)
    return (train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def cold_split(df: pd.DataFrame, cold_column: str, frac_valid=0.1, frac_test=0.2, seed=42):
    """Every id picked for valid/test is held out entirely -- never in train,
    and valid/test ids never overlap each other either."""
    rng = np.random.default_rng(seed)
    unique_ids = df[cold_column].unique().copy()
    rng.shuffle(unique_ids)

    n_test = max(1, int(len(unique_ids) * frac_test))
    n_valid = max(1, int(len(unique_ids) * frac_valid))
    test_ids = set(unique_ids[:n_test])
    valid_ids = set(unique_ids[n_test:n_test + n_valid])

    test_df = df[df[cold_column].isin(test_ids)]
    valid_df = df[df[cold_column].isin(valid_ids)]
    train_df = df[~df[cold_column].isin(test_ids) & ~df[cold_column].isin(valid_ids)]
    return (train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def cold_pair_split(df: pd.DataFrame, frac_valid=0.1, frac_test=0.2, seed=42):
    """Hardest level: valid/test pairs need BOTH drug and target unseen in train."""
    rng = np.random.default_rng(seed)
    drug_ids, target_ids = df["Drug_ID"].unique().copy(), df["Target_ID"].unique().copy()
    rng.shuffle(drug_ids)
    rng.shuffle(target_ids)

    def cut(ids, seed_offset):
        n_test = max(1, int(len(ids) * frac_test))
        n_valid = max(1, int(len(ids) * frac_valid))
        return set(ids[:n_test]), set(ids[n_test:n_test + n_valid])

    test_drugs, valid_drugs = cut(drug_ids, 0)
    test_targets, valid_targets = cut(target_ids, 1)

    is_test = df["Drug_ID"].isin(test_drugs) & df["Target_ID"].isin(test_targets)
    is_valid = df["Drug_ID"].isin(valid_drugs) & df["Target_ID"].isin(valid_targets)
    is_cold_drug = df["Drug_ID"].isin(test_drugs | valid_drugs)
    is_cold_target = df["Target_ID"].isin(test_targets | valid_targets)
    is_train = ~is_cold_drug & ~is_cold_target

    return (df[is_train].reset_index(drop=True),
            df[is_valid].reset_index(drop=True),
            df[is_test].reset_index(drop=True))


def build_all_splits(dataset_name: str) -> dict:
    df = load_deepdta_dataset(dataset_name)
    raw_splits = {
        "random": random_split(df),
        "cold_drug": cold_split(df, "Drug_ID"),
        "cold_target": cold_split(df, "Target_ID"),
        "cold_pair": cold_pair_split(df),
    }

    results = {}
    for split_name, (train_df, valid_df, test_df) in raw_splits.items():
        out_dir = os.path.join(SPLIT_DIR, dataset_name, split_name)
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        valid_df.to_csv(os.path.join(out_dir, "valid.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
        print(f"Saved {out_dir}/  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
        results[split_name] = {"train": train_df, "valid": valid_df, "test": test_df}
    return results


def check_no_leakage(splits: dict) -> None:
    for split_name, parts in splits.items():
        if split_name == "random":
            continue
        d_train, d_valid, d_test = (set(parts[p]["Drug_ID"]) for p in ("train", "valid", "test"))
        t_train, t_valid, t_test = (set(parts[p]["Target_ID"]) for p in ("train", "valid", "test"))
        if split_name in ("cold_drug", "cold_pair"):
            assert not (d_train & d_valid) and not (d_train & d_test) and not (d_valid & d_test), \
                f"LEAKAGE in {split_name} (drug)!"
        if split_name in ("cold_target", "cold_pair"):
            assert not (t_train & t_valid) and not (t_train & t_test) and not (t_valid & t_test), \
                f"LEAKAGE in {split_name} (target)!"
        print(f"{split_name}: no leakage across train/valid/test -- OK")


if __name__ == "__main__":
    for dataset_name in ("davis", "kiba"):
        print(f"\n=== Building splits for {dataset_name} ===")
        splits = build_all_splits(dataset_name)
        check_no_leakage(splits)