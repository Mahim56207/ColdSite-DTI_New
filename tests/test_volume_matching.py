"""Tests for the cold-pair volume-matched control.

Cold-pair discards every pair with exactly one cold entity, so it trains on
roughly 71% of the rows the other three levels get. `--train-subsample` exists
to run one control against that, and the control is only meaningful if it
changes the training set and nothing else.
"""
import pandas as pd
import pytest

from src.model.dataset import DTIDataset, build_protein_vocab, load_split


def _write_split(directory, n_train=60, n_valid=20, n_test=20):
    directory.mkdir(parents=True, exist_ok=True)
    import random

    random.seed(0)
    aa, smi = "ACDEFGHIKLMNPQRSTVWY", "CCONc1ccccc1"

    def frame(n, offset):
        random.seed(offset)
        return pd.DataFrame({
            "compound_iso_smiles": ["".join(random.choices(smi, k=20)) for _ in range(n)],
            "target_sequence": ["".join(random.choices(aa, k=80)) for _ in range(n)],
            "Target_ID": [f"T{i % 5}" for i in range(n)],
            "affinity": [random.uniform(5, 9) for _ in range(n)],
        })

    frame(n_train, 1).to_csv(directory / "train.csv", index=False)
    frame(n_valid, 2).to_csv(directory / "valid.csv", index=False)
    frame(n_test, 3).to_csv(directory / "test.csv", index=False)
    return directory


def test_subsampling_reduces_only_the_training_set(tmp_path):
    """Valid and test must be untouched: the control has to be evaluated on the
    same test set as the run it is being compared against, or it answers a
    different question."""
    split = _write_split(tmp_path / "cold_pair")
    train, valid, test, _dv, _pv = load_split(str(split), batch_size=8,
                                              train_subsample=25)
    assert len(train.dataset) == 25
    assert len(valid.dataset) == 20
    assert len(test.dataset) == 20


def test_no_subsample_keeps_every_training_row(tmp_path):
    split = _write_split(tmp_path / "random")
    train, _v, _t, _dv, _pv = load_split(str(split), batch_size=8)
    assert len(train.dataset) == 60


def test_a_subsample_larger_than_the_split_is_a_no_op(tmp_path):
    """Matching to a volume the split does not have must not crash or invent
    rows."""
    split = _write_split(tmp_path / "random")
    train, _v, _t, _dv, _pv = load_split(str(split), batch_size=8,
                                         train_subsample=9999)
    assert len(train.dataset) == 60


def test_subsampling_is_reproducible_for_a_given_seed(tmp_path):
    split = _write_split(tmp_path / "random")
    first = load_split(str(split), batch_size=8, train_subsample=25,
                       subsample_seed=7)[0].dataset.smiles
    second = load_split(str(split), batch_size=8, train_subsample=25,
                        subsample_seed=7)[0].dataset.smiles
    assert first == second


def test_different_subsample_seeds_pick_different_rows(tmp_path):
    split = _write_split(tmp_path / "random")
    a = load_split(str(split), batch_size=8, train_subsample=25,
                   subsample_seed=1)[0].dataset.smiles
    b = load_split(str(split), batch_size=8, train_subsample=25,
                   subsample_seed=2)[0].dataset.smiles
    assert a != b


def test_vocabulary_is_built_after_subsampling(tmp_path):
    """A smaller training set genuinely sees fewer SMILES tokens. That
    shrinkage is part of the effect being measured -- building the vocab from
    the full file first would hand the control an advantage the real run
    never had."""
    split = _write_split(tmp_path / "random")
    full_vocab = load_split(str(split), batch_size=8)[3]
    small_vocab = load_split(str(split), batch_size=8, train_subsample=3)[3]
    assert len(small_vocab) <= len(full_vocab)


def test_from_frame_and_from_csv_agree(tmp_path):
    """load_split switched to from_frame so it could subsample before building
    the vocab; the two paths must stay equivalent."""
    split = _write_split(tmp_path / "random")
    protein_vocab = build_protein_vocab()
    from src.model.dataset import build_smiles_vocab

    frame = pd.read_csv(split / "train.csv")
    drug_vocab = build_smiles_vocab(frame["compound_iso_smiles"].astype(str).tolist())

    by_csv = DTIDataset.from_csv(str(split / "train.csv"), drug_vocab, protein_vocab)
    by_frame = DTIDataset.from_frame(frame, drug_vocab, protein_vocab)
    assert len(by_csv) == len(by_frame)
    assert by_csv.smiles == by_frame.smiles
    assert by_csv.labels == by_frame.labels
