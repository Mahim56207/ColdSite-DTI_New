"""Track A split tests.

docs/01_GUIDE_124AD0008.md: 'If it fails, the whole paper's core claim is
compromised -- so check it now, not in October.' The leakage check lives in
build_splits.py; these tests check the checker, on synthetic data, so they run
without DAVIS/KIBA present.
"""
import pandas as pd
import pytest

from src.data.build_splits import check_no_leakage, cold_pair_split, cold_split, random_split


def toy_df(n_drugs=20, n_targets=15):
    rows = [
        {"Drug_ID": f"D{d}", "Drug": "CCO", "Target_ID": f"T{t}",
         "Target": "MSTNPKPQ", "Y": float(d + t)}
        for d in range(n_drugs) for t in range(n_targets)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# cold-drug / cold-target
# --------------------------------------------------------------------------

def test_cold_drug_split_shares_no_drug_between_train_and_test():
    train, valid, test = cold_split(toy_df(), "Drug_ID")
    assert not set(train["Drug_ID"]) & set(test["Drug_ID"])
    assert not set(train["Drug_ID"]) & set(valid["Drug_ID"])
    assert not set(valid["Drug_ID"]) & set(test["Drug_ID"])


def test_cold_drug_split_still_shares_targets():
    """Cold-DRUG means the protein is familiar -- that is the point of level 2."""
    train, _valid, test = cold_split(toy_df(), "Drug_ID")
    assert set(train["Target_ID"]) & set(test["Target_ID"])


def test_cold_target_split_shares_no_target():
    train, valid, test = cold_split(toy_df(), "Target_ID")
    assert not set(train["Target_ID"]) & set(test["Target_ID"])
    assert not set(train["Target_ID"]) & set(valid["Target_ID"])
    assert not set(valid["Target_ID"]) & set(test["Target_ID"])


def test_cold_split_loses_no_rows_it_should_not():
    df = toy_df()
    train, valid, test = cold_split(df, "Drug_ID")
    assert len(train) + len(valid) + len(test) == len(df)


def test_cold_split_is_deterministic_for_a_seed():
    a = cold_split(toy_df(), "Drug_ID", seed=7)[2]["Drug_ID"].tolist()
    b = cold_split(toy_df(), "Drug_ID", seed=7)[2]["Drug_ID"].tolist()
    assert a == b


def test_different_seeds_give_different_held_out_drugs():
    a = set(cold_split(toy_df(), "Drug_ID", seed=1)[2]["Drug_ID"])
    b = set(cold_split(toy_df(), "Drug_ID", seed=2)[2]["Drug_ID"])
    assert a != b


def test_no_split_is_empty():
    for part in cold_split(toy_df(), "Drug_ID"):
        assert len(part) > 0


# --------------------------------------------------------------------------
# cold-pair (the hardest level)
# --------------------------------------------------------------------------

def test_cold_pair_holds_out_both_axes():
    train, valid, test = cold_pair_split(toy_df(30, 25))
    assert not set(train["Drug_ID"]) & set(test["Drug_ID"])
    assert not set(train["Target_ID"]) & set(test["Target_ID"])
    assert not set(train["Drug_ID"]) & set(valid["Drug_ID"])
    assert not set(train["Target_ID"]) & set(valid["Target_ID"])


def test_cold_pair_test_rows_are_new_on_both_axes_simultaneously():
    """Level 4 means BOTH unseen in the same row, not either-or."""
    train, _valid, test = cold_pair_split(toy_df(30, 25))
    train_drugs, train_targets = set(train["Drug_ID"]), set(train["Target_ID"])
    for _, row in test.iterrows():
        assert row["Drug_ID"] not in train_drugs
        assert row["Target_ID"] not in train_targets


def test_cold_pair_discards_the_off_diagonal_rows():
    """Rows with a held-out drug but a training target belong to no split.

    This is expected, not a bug -- but it means cold-pair trains on far less
    data than the other levels, which has to be stated when the accuracy drop
    at level 4 is interpreted.
    """
    df = toy_df(30, 25)
    train, valid, test = cold_pair_split(df)
    assert len(train) + len(valid) + len(test) < len(df)


# --------------------------------------------------------------------------
# random / warm
# --------------------------------------------------------------------------

def test_random_split_covers_every_row_exactly_once():
    df = toy_df()
    train, valid, test = random_split(df)
    assert len(train) + len(valid) + len(test) == len(df)


def test_random_split_proportions_are_roughly_right():
    df = toy_df(40, 40)
    train, valid, test = random_split(df, frac_valid=0.1, frac_test=0.2)
    assert len(test) / len(df) == pytest.approx(0.2, abs=0.02)
    assert len(valid) / len(df) == pytest.approx(0.1, abs=0.02)


def test_warm_split_deliberately_shares_ids():
    """Level 1 is supposed to be easy -- that is what it is a control for."""
    train, _valid, test = random_split(toy_df())
    assert set(train["Drug_ID"]) & set(test["Drug_ID"])
    assert set(train["Target_ID"]) & set(test["Target_ID"])


# --------------------------------------------------------------------------
# the leakage checker itself
# --------------------------------------------------------------------------

def _as_dict(name, parts):
    return {name: {"train": parts[0], "valid": parts[1], "test": parts[2]}}


def test_checker_passes_on_correctly_built_splits():
    df = toy_df()
    check_no_leakage(_as_dict("cold_drug", cold_split(df, "Drug_ID")))
    check_no_leakage(_as_dict("cold_target", cold_split(df, "Target_ID")))
    check_no_leakage(_as_dict("cold_pair", cold_pair_split(df)))


def test_checker_catches_injected_drug_leakage():
    df = toy_df()
    train, valid, test = cold_split(df, "Drug_ID")
    leaked = pd.concat([train, test.head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_no_leakage(_as_dict("cold_drug", (leaked, valid, test)))


def test_checker_catches_injected_target_leakage():
    df = toy_df()
    train, valid, test = cold_split(df, "Target_ID")
    leaked = pd.concat([train, test.head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_no_leakage(_as_dict("cold_target", (leaked, valid, test)))


def test_checker_catches_valid_test_overlap():
    df = toy_df()
    train, valid, test = cold_split(df, "Drug_ID")
    contaminated = pd.concat([valid, test.head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_no_leakage(_as_dict("cold_drug", (train, contaminated, test)))


def test_checker_deliberately_skips_the_warm_split():
    """The random split is supposed to share IDs; flagging it would be wrong."""
    check_no_leakage(_as_dict("random", random_split(toy_df())))
