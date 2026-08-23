"""Tests for the minimum-epoch floor on checkpoint selection.

The floor exists because of a measured problem, not a hypothetical one: three
seeds of DeepDTA on DAVIS took their cold_pair checkpoint at epoch 2.3 on
average, against 15.7 for random, because cold_pair validation is 264 rows and
its validation loss bottoms out immediately. `run_ladder` and `run_audit` then
read attention out of those epoch-2 weights, which would have made "explanations
degrade across the ladder" partly a statement about training epochs.

What must hold:
  * no checkpoint, and no early stop, before the floor
  * a run shorter than the floor still produces a checkpoint
  * min_epochs=1 reproduces the old behaviour exactly, so the pre-floor numbers
    remain reproducible from this code
  * all three trainers use this one implementation
"""
import pytest

from src.model.early_stopping import DEFAULT_MIN_EPOCHS, CheckpointSelector


def run(losses, patience=10, min_epochs=DEFAULT_MIN_EPOCHS, n_epochs=None):
    """Drive a selector over a loss curve. Returns (saved_epochs, stopped_at)."""
    selector = CheckpointSelector(patience=patience, min_epochs=min_epochs,
                                  n_epochs=n_epochs or len(losses))
    saved, stopped_at = [], None
    for epoch, loss in enumerate(losses, start=1):
        if selector.consider(epoch, loss):
            saved.append(epoch)
        elif selector.should_stop(epoch):
            stopped_at = epoch
            break
    return saved, stopped_at, selector


def test_nothing_is_saved_before_the_floor():
    """The cold_pair failure mode: loss is lowest at epoch 2 and the run would
    have checkpointed there."""
    losses = [0.30, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]
    saved, _stopped, selector = run(losses, min_epochs=10)

    assert 2 not in saved
    assert saved == [10]
    assert selector.best_epoch == 10


def test_early_stopping_cannot_fire_before_the_floor():
    """Without the guard the patience counter runs against best_epoch = -1 and
    stops the run before a single checkpoint exists."""
    losses = [0.5] * 30
    saved, stopped, _selector = run(losses, patience=5, min_epochs=10)

    assert saved == [10], "first checkpoint should be taken at the floor"
    assert stopped is not None and stopped >= 15


def test_a_well_behaved_curve_is_unaffected():
    """random and cold_target already settle after the floor, so their
    checkpoints must not move."""
    losses = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25,
              0.20, 0.18, 0.17, 0.16, 0.15, 0.16, 0.17]
    floored, _s1, sel_floored = run(losses, min_epochs=10)
    unfloored, _s2, sel_unfloored = run(losses, min_epochs=1)

    # the minimum is at epoch 15, well past the floor, so both agree
    assert sel_floored.best_epoch == sel_unfloored.best_epoch == 15
    assert floored[-1] == unfloored[-1] == 15


def test_min_epochs_one_reproduces_the_unfloored_behaviour():
    """The pre-floor results have to stay reproducible from this code, or the
    change is not auditable."""
    losses = [0.30, 0.22, 0.24, 0.26, 0.28]
    saved, _stopped, selector = run(losses, min_epochs=1)
    assert saved == [1, 2]
    assert selector.best_epoch == 2


def test_a_run_shorter_than_the_floor_still_checkpoints():
    """A smoke test with --epochs 2 must not silently produce no checkpoint;
    the cell would then look untrained rather than fail."""
    losses = [0.4, 0.3]
    saved, _stopped, selector = run(losses, min_epochs=10, n_epochs=2)

    assert saved, "a 2-epoch run with a floor of 10 still needs a checkpoint"
    assert selector.best_epoch == 2
    assert selector.summary()["min_epochs_applied"] == 2
    assert selector.summary()["min_epochs"] == 10


def test_summary_reports_what_methods_has_to_state():
    losses = [0.5, 0.4, 0.3, 0.2, 0.1] * 3
    _saved, _stopped, selector = run(losses, min_epochs=10)
    summary = selector.summary()

    assert set(summary) == {"best_epoch", "best_val_loss", "min_epochs",
                            "min_epochs_applied"}
    assert summary["best_epoch"] >= 10
    assert summary["min_epochs"] == 10


@pytest.mark.parametrize("kwargs", [{"patience": 0}, {"min_epochs": 0}])
def test_nonsense_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        CheckpointSelector(**{"patience": 10, "min_epochs": 10, **kwargs})


def test_all_three_trainers_share_this_implementation():
    """The audit compares models to each other, so the floor has to be the same
    object in all three -- three hand-written copies are three things that can
    drift, and the loops are not even indexed alike."""
    from src.model import train, train_deepdta, train_hyperattentiondti

    for module in (train, train_deepdta, train_hyperattentiondti):
        assert module.CheckpointSelector is CheckpointSelector, module.__name__
        assert module.DEFAULT_MIN_EPOCHS == DEFAULT_MIN_EPOCHS, module.__name__
