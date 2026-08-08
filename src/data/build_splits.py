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
    # np.asarray(..., dtype=object) is not cosmetic: df[col].unique() returns a
    # pandas StringArray on string-dtype columns, and numpy's shuffle warns that
    # it cannot guarantee correct behaviour on non-Sequence array objects -- it
    # can leave duplicates behind. A duplicated ID here would put the same drug
    # in train and test and silently void the paper's core claim.
    unique_ids = np.asarray(df[cold_column].unique(), dtype=object)
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
    # dtype=object for the same reason as in cold_split -- see the note there.
    drug_ids = np.asarray(df["Drug_ID"].unique(), dtype=object)
    target_ids = np.asarray(df["Target_ID"].unique(), dtype=object)
    rng.shuffle(drug_ids)
    rng.shuffle(target_ids)

    def cut(ids):
        n_test = max(1, int(len(ids) * frac_test))
        n_valid = max(1, int(len(ids) * frac_valid))
        return set(ids[:n_test]), set(ids[n_test:n_test + n_valid])

    test_drugs, valid_drugs = cut(drug_ids)
    test_targets, valid_targets = cut(target_ids)

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


def summarise_splits(splits: dict, dataset_name: str) -> pd.DataFrame:
    """Per-split drug/protein/pair counts -- docs/01_GUIDE_124AD0008.md Step 2.

    Worth reading before any training run, not just filing. Cold-pair discards
    every row whose drug is held out but whose target is not, so its train set
    is dramatically smaller than the other three levels. If that is not on the
    table, the accuracy drop at level 4 gets attributed entirely to difficulty
    when part of it is simply less training data.
    """
    rows = []
    for split_name, parts in splits.items():
        for part_name in ("train", "valid", "test"):
            part = parts[part_name]
            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "part": part_name,
                "pairs": len(part),
                "drugs": part["Drug_ID"].nunique(),
                "proteins": part["Target_ID"].nunique(),
            })
    summary = pd.DataFrame(rows)

    # TWO different ratios, and conflating them is the mistake this function
    # exists to prevent. Cold-pair is smaller in two distinct ways:
    #
    #   pct_of_all_pairs_used  train+valid+test as a share of the largest
    #                          split's total -- depressed because cold-pair
    #                          DISCARDS every row with exactly one cold entity
    #                          (~54% for DAVIS)
    #   train_pct_of_largest   TRAIN rows as a share of the largest split's
    #                          train rows -- the training-volume confound
    #                          itself (~71% for DAVIS)
    #
    # The old single column was named `pct_of_largest_split` but computed the
    # first, and it was repeatedly quoted as if it were the second. That
    # overstates the training confound by roughly a factor of two.
    totals = summary.groupby("split")["pairs"].sum()
    summary["pct_of_all_pairs_used"] = summary["split"].map(
        lambda s: round(100 * totals[s] / totals.max(), 1)
    )

    train_rows = summary[summary["part"] == "train"].set_index("split")["pairs"]
    summary["train_pct_of_largest"] = summary["split"].map(
        lambda s: round(100 * train_rows[s] / train_rows.max(), 1)
    )

    # kept so existing readers do not break; it is the "used" ratio, as before
    summary["pct_of_largest_split"] = summary["pct_of_all_pairs_used"]
    return summary


def write_split_report(summaries: list, out_path: str = "results/split_summary.md") -> None:
    """Write the team-facing table."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    combined = pd.concat(summaries, ignore_index=True)
    with open(out_path, "w") as f:
        f.write("# Split summary\n\n")
        f.write("Generated by `python -m src.data.build_splits`. "
                "Do not edit by hand.\n\n")
        f.write(combined.to_markdown(index=False))
        f.write("\n")
    print(f"\nSaved split summary -> {out_path}")


if __name__ == "__main__":
    summaries = []
    for dataset_name in ("davis", "kiba"):
        print(f"\n=== Building splits for {dataset_name} ===")
        splits = build_all_splits(dataset_name)
        check_no_leakage(splits)
        summaries.append(summarise_splits(splits, dataset_name))
    write_split_report(summaries)