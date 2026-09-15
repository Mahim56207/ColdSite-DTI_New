"""A diverged cell must stop at once and say why.

2026-09-14, KIBA under `--amp`: MolTrans produced NaN from batch ~4,040 of epoch 1. Its
hand-rolled LayerNorm divides by `sqrt(var + 1e-12)`, and 1e-12 is below float16's
smallest subnormal, so a zero-variance row gives 0/0. The run then finished the epoch and
740 validation batches before failing inside sklearn with "Input contains NaN" --
seventeen minutes lost per cell, for a message naming neither the model nor the cause.

These tests pin both guards: the one in MolTrans's training loop that stops at the batch,
and the one in `compute_metrics` that every trainer's metrics have to pass through. They
also pin the thing that matters more than either: a healthy cell must be scored exactly
as it was before the guards existed.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.train import compute_metrics


# ---------------------------------------------------------------------------
# compute_metrics: the central guard, shared by all four trainers
# ---------------------------------------------------------------------------

def test_a_healthy_cell_is_scored_exactly_as_before():
    y = [0, 1, 0, 1, 1, 0]
    logits = [-2.0, 1.5, -0.5, 0.8, 2.2, -1.1]      # separates the classes perfectly
    m = compute_metrics(y, logits, "binary")
    assert m["auroc"] == pytest.approx(1.0)
    assert m["auprc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)

    # and an imperfect one, so the test would notice a guard that clamped scores
    swapped = [-2.0, 1.5, 0.9, 0.8, 2.2, -1.1]      # row 2 is a negative scored positive
    worse = compute_metrics(y, swapped, "binary")
    assert worse["auroc"] == pytest.approx(8 / 9)
    assert worse["accuracy"] == pytest.approx(5 / 6)


def test_nan_predictions_are_refused_with_a_message_that_names_the_cause():
    y = [0, 1, 0, 1]
    logits = [0.5, float("nan"), -0.5, 1.0]
    with pytest.raises(ValueError, match="not finite"):
        compute_metrics(y, logits, "binary")
    with pytest.raises(ValueError, match="full precision"):
        compute_metrics(y, logits, "binary")


def test_infinite_predictions_are_refused_too():
    for bad in (float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            compute_metrics([0, 1, 0], [0.1, bad, -0.2], "binary")


def test_the_message_counts_how_many_predictions_went_bad():
    logits = [float("nan")] * 3 + [0.1, 0.2]
    with pytest.raises(ValueError, match="3 of 5 predictions"):
        compute_metrics([0, 1, 0, 1, 0], logits, "binary")


def test_regression_predictions_are_guarded_as_well():
    with pytest.raises(ValueError, match="not finite"):
        compute_metrics([1.0, 2.0, 3.0], [1.0, float("nan"), 3.0], "regression")


def test_a_one_class_validation_batch_still_returns_nan_metrics_rather_than_raising():
    """The pre-existing behaviour: AUROC is undefined on one class, and an overnight run
    must not die of it. That NaN is in the METRIC, not in the predictions, so the guard
    must not catch it."""
    m = compute_metrics([1, 1, 1], [0.2, 0.9, 0.4], "binary")
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"])
    assert not np.isnan(m["accuracy"])


# ---------------------------------------------------------------------------
# MolTrans's training loop: stop at the batch, not at the end of the epoch
# ---------------------------------------------------------------------------

class Diverging(nn.Module):
    """Finite for the first few batches, then NaN -- the shape of the real failure."""

    def __init__(self, fail_at=3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.batch_size = 2
        self.calls = 0
        self.fail_at = fail_at

    def forward(self, d, p, dm, pm):
        self.calls += 1
        value = float("nan") if self.calls >= self.fail_at else 0.0
        return torch.full((d.shape[0], 1), value) + self.weight * 0


def _loader(n_batches=8, batch=2):
    rows = []
    for _ in range(n_batches):
        rows.append((torch.zeros(batch, 4, dtype=torch.long),
                     torch.zeros(batch, 4, dtype=torch.long),
                     torch.ones(batch, 4), torch.ones(batch, 4),
                     torch.zeros(batch)))
    return rows


def test_the_loop_stops_at_the_batch_that_diverges():
    from src.model import precision
    from src.model.train_moltrans import run_epoch

    model = Diverging(fail_at=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = precision.make_scaler("cpu", False)
    with pytest.raises(ValueError, match="loss is nan"):
        run_epoch(model, _loader(n_batches=8), nn.BCEWithLogitsLoss(), "cpu",
                  optimizer=optimizer, scaler=scaler, log_every=0)
    # 8 batches were available; it must not have run them all.
    assert model.calls == 3, f"ran {model.calls} batches, should have stopped at 3"


def test_the_message_tells_you_to_drop_amp_and_which_model_it_is():
    from src.model import precision
    from src.model.train_moltrans import run_epoch

    model = Diverging(fail_at=1)
    with pytest.raises(ValueError) as excinfo:
        run_epoch(model, _loader(n_batches=4), nn.BCEWithLogitsLoss(), "cpu",
                  optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
                  scaler=precision.make_scaler("cpu", False), log_every=0)
    message = str(excinfo.value)
    assert "diverged" in message
    assert "--amp" in message
    assert "LayerNorm" in message


def test_a_finite_epoch_is_unchanged_by_the_guard():
    """The loss is read once and summed the same way, so a healthy epoch returns exactly
    what it returned before -- the guard must cost nothing but the comparison."""
    from src.model import precision
    from src.model.train_moltrans import run_epoch

    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.batch_size = 2

        def forward(self, d, p, dm, pm):
            return torch.zeros(d.shape[0], 1) + self.weight

    model = Flat()
    loss, trues, scores = run_epoch(model, _loader(n_batches=5), nn.BCEWithLogitsLoss(),
                                    "cpu", optimizer=None,
                                    scaler=precision.make_scaler("cpu", False), log_every=0)
    assert np.isfinite(loss)
    # BCEWithLogitsLoss(0, 0) = log(2)
    assert loss == pytest.approx(float(np.log(2)), rel=1e-5)
    assert len(trues) == 10 and len(scores) == 10
