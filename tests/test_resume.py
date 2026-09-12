"""Tests for epoch-level resume (src/model/resume.py) and mixed precision
(src/model/precision.py).

A KIBA cell can outlast one 11-hour Kaggle commit. What has to hold:

* a run stopped after any epoch and restarted ends exactly where the
  uninterrupted run ends (on a CPU, bit for bit);
* the saved best checkpoint agrees with the selector whatever instant the
  kill lands at;
* a cell cannot be continued under different settings;
* nothing downstream mistakes a resume file for a trained cell;
* with --amp off, the training step is the old one.

The per-trainer versions of the first point (all four trainers, stopped after
epoch 2 of 4) were run by hand on real DAVIS rows before this was committed;
here the same loop is driven on a toy model so it runs in seconds.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model import precision, resume
from src.model.checkpoint_naming import checkpoint_path, discover_checkpoints
from src.model.early_stopping import CheckpointSelector


# --------------------------------------------------------------------------
# a toy trainer with the same loop shape as the four real ones
# --------------------------------------------------------------------------

SETTINGS = {"amp": False, "lr": 0.05, "epochs": 6}
KEYS = tuple(SETTINGS)


def _data():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 5, generator=generator)
    y = x @ torch.arange(1.0, 6.0) + 0.1 * torch.randn(64, generator=generator)
    return x, y


def _train(ckpt, settings=SETTINGS, stop_after=None, n_epochs=6, patience=10):
    """Train until done or `stop_after`; returns (resumable, params, history)."""
    torch.manual_seed(1)
    np.random.seed(1)
    x, y = _data()
    model = nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    selector = CheckpointSelector(patience=patience, min_epochs=1, n_epochs=n_epochs)
    scaler = precision.make_scaler("cpu", settings["amp"])
    run = resume.Resumable(ckpt, "cpu", settings, KEYS, stop_after)
    start = run.begin(model, optimizer, selector, scheduler=scheduler, scaler=scaler)
    history = run.extra.get("history", [])

    for epoch in (range(start, n_epochs + 1) if start else ()):
        # batch order drawn from the global RNG, like a shuffling DataLoader, so
        # a resume that failed to restore the RNG would train on other batches
        for idx in torch.randperm(64).split(16):
            optimizer.zero_grad()
            loss = ((model(x[idx]).squeeze(-1) - y[idx]) ** 2).mean()
            scaler.scale(loss).backward()
            precision.step(optimizer, scaler, model.parameters(), clip=5.0)
        with torch.no_grad():
            val_loss = float(((model(x).squeeze(-1) - y) ** 2).mean())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "val_loss": val_loss})

        best = ({"model_state": model.state_dict(), "epoch": epoch}
                if selector.consider(epoch, val_loss) else None)
        stopping = best is None and selector.should_stop(epoch)
        run.end_epoch(epoch, finished=stopping or epoch == n_epochs,
                      best_checkpoint=best, history=history)
        if stopping:
            break
        if run.interrupt_now(epoch):
            return run, None, history
    params = [p.detach().clone() for p in model.parameters()]
    return run, params, history


def _ckpt(tmp_path, name="a"):
    return str(tmp_path / name / "coldsite_dti_davis_random_binary_seed1.pt")


def _prepare(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# --------------------------------------------------------------------------
# continuing is exact
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stop_after", [1, 3, 5])
def test_a_resumed_run_ends_exactly_where_the_uninterrupted_one_does(tmp_path, stop_after):
    whole = _prepare(_ckpt(tmp_path, "whole"))
    split = _prepare(_ckpt(tmp_path, "split"))

    _, params_whole, history_whole = _train(whole)
    run, params, _ = _train(split, stop_after=stop_after)
    assert params is None and os.path.exists(run.path)
    run, params_split, history_split = _train(split)

    assert run.resumed_from == stop_after
    assert history_split == history_whole
    for a, b in zip(params_whole, params_split):
        assert torch.equal(a, b)
    best_whole = torch.load(whole, weights_only=False)
    best_split = torch.load(split, weights_only=False)
    assert best_whole["epoch"] == best_split["epoch"]
    for key in best_whole["model_state"]:
        assert torch.equal(best_whole["model_state"][key], best_split["model_state"][key])


def test_a_fresh_cell_starts_at_epoch_one(tmp_path):
    run = resume.Resumable(_prepare(_ckpt(tmp_path)), "cpu", SETTINGS, KEYS)
    model = nn.Linear(2, 1)
    start = run.begin(model, torch.optim.SGD(model.parameters(), lr=0.1),
                      CheckpointSelector(patience=3, min_epochs=1))
    assert start == 1 and run.resumed_from is None and run.extra == {}


def test_a_cell_that_finished_training_goes_straight_to_the_test_pass(tmp_path):
    """Killed after early stopping, before the results were written."""
    ckpt = _prepare(_ckpt(tmp_path))
    _train(ckpt)                                   # runs to the end, never cleared
    run = resume.Resumable(ckpt, "cpu", SETTINGS, KEYS)
    model = nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters())
    assert run.begin(model, optimizer, CheckpointSelector(patience=10, min_epochs=1)) is None


def test_the_resume_file_is_removed_once_the_cell_is_done(tmp_path):
    ckpt = _prepare(_ckpt(tmp_path))
    run, _, _ = _train(ckpt)
    assert os.path.exists(run.path)
    run.clear()
    assert not os.path.exists(run.path)
    assert os.path.exists(ckpt), "clearing the resume file must keep the checkpoint"


def test_the_save_is_atomic(tmp_path):
    path = str(tmp_path / "x_resume.pt")
    resume.save(path, {"a": 1})
    assert torch.load(path)["a"] == 1
    assert not os.path.exists(path + ".tmp")


# --------------------------------------------------------------------------
# the checkpoint agrees with the selector whenever the kill lands
# --------------------------------------------------------------------------

def test_a_kill_between_the_two_saves_is_repaired(tmp_path):
    """end_epoch writes the resume file, then the checkpoint. Simulate a kill
    in between: the resume file names the new best epoch, the checkpoint on
    disk is still the previous best."""
    ckpt = _prepare(_ckpt(tmp_path))
    run, _, _ = _train(ckpt, stop_after=3)
    state = torch.load(run.path, weights_only=False)
    meta = state["best_checkpoint_meta"]
    if meta is None:
        pytest.skip("epoch 3 was not a best epoch in this toy run")
    torch.save({"model_state": {k: torch.zeros_like(v) for k, v in state["model_state"].items()},
                "epoch": meta["epoch"] - 1}, ckpt)

    _train(ckpt, n_epochs=3)                        # resumes, repairs, trains nothing
    repaired = torch.load(ckpt, weights_only=False)
    assert repaired["epoch"] == meta["epoch"]
    for key, value in state["model_state"].items():
        assert torch.equal(repaired["model_state"][key], value)


def test_a_consistent_checkpoint_is_left_alone(tmp_path):
    ckpt = _prepare(_ckpt(tmp_path))
    _train(ckpt, stop_after=2)
    before = os.path.getmtime(ckpt)
    run = resume.Resumable(ckpt, "cpu", SETTINGS, KEYS)
    run._repair_checkpoint(torch.load(run.path, weights_only=False))
    assert os.path.getmtime(ckpt) == before


# --------------------------------------------------------------------------
# one cell, one protocol
# --------------------------------------------------------------------------

def test_resuming_under_different_settings_is_refused(tmp_path):
    ckpt = _prepare(_ckpt(tmp_path))
    _train(ckpt, stop_after=2)
    with pytest.raises(SystemExit, match="amp: was False, now True"):
        _train(ckpt, settings={**SETTINGS, "amp": True})


def test_settings_outside_the_keys_do_not_block_a_resume(tmp_path):
    """--stop-after-epoch itself differs between the two halves of the test run."""
    resume.check_compatible({"lr": 1, "stop_after_epoch": 2},
                            {"lr": 1, "stop_after_epoch": None}, ("lr",))


def test_the_selector_state_round_trips():
    a = CheckpointSelector(patience=3, min_epochs=2, n_epochs=10)
    for epoch, loss in enumerate([5.0, 4.0, 3.0, 3.5], start=1):
        a.consider(epoch, loss)
    b = CheckpointSelector(patience=99, min_epochs=1)
    b.load_state_dict(a.state_dict())
    assert b.summary() == a.summary()
    assert [b.should_stop(e) for e in (5, 6)] == [a.should_stop(e) for e in (5, 6)]


# --------------------------------------------------------------------------
# nothing downstream reads a resume file as a trained cell
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["coldsite_dti", "deepdta", "hyperattentiondti", "moltrans"])
def test_discovery_ignores_resume_files(tmp_path, model):
    ckpt = checkpoint_path(str(tmp_path), "davis", "random", "binary", 1, model=model)
    resume.save(resume.resume_path(ckpt), {"epoch": 3})
    for other in ("coldsite_dti", "deepdta", "hyperattentiondti", "moltrans"):
        assert discover_checkpoints(str(tmp_path), model=other) == []
    torch.save({"model_state": {}, "epoch": 3}, ckpt)
    assert len(discover_checkpoints(str(tmp_path), model=model)) == 1


def test_the_resume_path_sits_beside_the_checkpoint():
    assert resume.resume_path("r/x_seed1.pt") == "r/x_seed1_resume.pt"
    with pytest.raises(ValueError):
        resume.resume_path("r/x_seed1.json")


# --------------------------------------------------------------------------
# mixed precision
# --------------------------------------------------------------------------

def test_amp_off_is_the_plain_step():
    """A disabled GradScaler must reduce to backward / clip / step exactly."""
    torch.manual_seed(0)
    x, y = _data()
    reference, candidate = nn.Linear(5, 1), nn.Linear(5, 1)
    candidate.load_state_dict(reference.state_dict())
    opt_ref = torch.optim.Adam(reference.parameters(), lr=0.1)
    opt_new = torch.optim.Adam(candidate.parameters(), lr=0.1)
    scaler = precision.make_scaler("cpu", False)
    for _ in range(5):
        opt_ref.zero_grad()
        ((reference(x).squeeze(-1) - y) ** 2).mean().backward()
        nn.utils.clip_grad_norm_(reference.parameters(), max_norm=5.0)
        opt_ref.step()

        opt_new.zero_grad()
        with precision.autocast("cpu", False):
            loss = ((candidate(x).squeeze(-1) - y) ** 2).mean()
        scaler.scale(loss).backward()
        precision.step(opt_new, scaler, candidate.parameters(), clip=5.0)
    for a, b in zip(reference.parameters(), candidate.parameters()):
        assert torch.equal(a, b)


def test_the_scaler_is_only_live_on_a_gpu():
    assert not precision.make_scaler("cpu", True).is_enabled()
    assert not precision.make_scaler("cuda", False).is_enabled()


def test_amp_on_a_cpu_runs_in_bfloat16():
    with precision.autocast("cpu", True):
        out = nn.Linear(4, 2)(torch.randn(3, 4))
    assert out.dtype == torch.bfloat16


def test_amp_trains_coldsite_dti_on_a_cpu():
    """The one real model exercised here: its BiLSTM and attention run under
    autocast, and the loss stays finite."""
    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.dataset import make_loader, random_dataset
    from src.model.train import evaluate, train_one_epoch

    torch.manual_seed(0)
    data = random_dataset(32, binary=True, seed=1)
    loader = make_loader(data, batch_size=16, shuffle=True)
    model = ColdSiteDTI(len(data.drug_vocab) + 2, len(data.protein_vocab) + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loss = train_one_epoch(model, loader, optimizer, loss_fn, "cpu", amp=True)
    val_loss, metrics = evaluate(model, loader, loss_fn, "cpu", "binary", amp=True)
    assert np.isfinite(train_loss) and np.isfinite(val_loss)
    assert set(metrics) == {"auroc", "auprc", "accuracy"}


# --------------------------------------------------------------------------
# the Kaggle notebook passes the settings through
# --------------------------------------------------------------------------

NOTEBOOK = "notebooks/kaggle_binary_grid.ipynb"


def _notebook_commands(amp, models, kiba_run=None):
    cells = json.load(open(NOTEBOOK))["cells"]
    settings = next(c for c in cells if "SETTINGS" in "".join(c["source"]))
    runner = next(c for c in cells if "def build_queue" in "".join(c["source"]))
    source = "".join(settings["source"])
    source = source.replace("AMP = False", f"AMP = {amp}")
    source = source.replace("MODELS = ['moltrans']", f"MODELS = {models!r}")
    source = source.replace("KIBA_RUN = None", f"KIBA_RUN = {kiba_run!r}")
    import time
    namespace = {"time": time, "os": os, "START": 0.0, "N_GPU": 2, "WORK": "/w", "RESULTS": "/w/results",
                 "COLDSITE_BATCH": 16, "HAT_BATCH": 8, "HAT_ACCUM": 4,
                 "DEEPDTA_BATCH": 256, "MOLTRANS_BATCH": 16}
    exec(source, namespace)
    exec("".join(runner["source"]), namespace)
    return [cmd for splits in namespace["SHARE"].values()
            for cmd in namespace["build_queue"](splits)]


ALL_MODELS = ["deepdta", "coldsite_dti", "hyperattentiondti", "moltrans"]


@pytest.mark.parametrize("amp", [True, False])
def test_the_notebook_gives_every_cell_the_same_amp_setting(amp):
    commands = _notebook_commands(amp, ALL_MODELS)
    assert {cmd[3] for cmd in commands} == {
        "src.model.train_deepdta", "src.model.run_grid",
        "src.model.train_hyperattentiondti", "src.model.train_moltrans"}
    assert all(("--amp" in cmd) == amp for cmd in commands)


def _cell_of(cmd):
    """(model, split, seed) for a per-cell command; run_grid expands to several."""
    arg = lambda flag: cmd[cmd.index(flag) + 1]
    model = {"src.model.train_deepdta": "deepdta", "src.model.train_moltrans": "moltrans",
             "src.model.train_hyperattentiondti": "hyperattentiondti",
             "src.model.run_grid": "coldsite_dti"}[cmd[3]]
    if model == "coldsite_dti":
        return [(model, split, int(seed)) for split in arg("--splits").split(",")
                for seed in arg("--seeds").split(",")]
    return [(model, arg("--split"), int(arg("--seed")))]


def test_the_six_kiba_runs_are_the_48_cells_each_exactly_once():
    """Six accounts, one KIBA_RUN each: nothing trained twice, nothing left out."""
    seen = []
    for run in ("K1", "K2", "K3", "K4", "K5", "K6"):
        commands = _notebook_commands(False, ["moltrans"], kiba_run=run)
        assert all("--amp" in cmd for cmd in commands), f"{run} must train with --amp"
        assert all(("--datasets" if cmd[3] == "src.model.run_grid" else "--dataset") in cmd
                   and "kiba" in cmd for cmd in commands), f"{run} must train KIBA"
        seen += [cell for cmd in commands for cell in _cell_of(cmd)]
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(m, s, seed) for m in ALL_MODELS
                         for s in ("random", "cold_drug", "cold_target", "cold_pair")
                         for seed in (1, 2, 3)}


def test_the_notebook_clones_the_configured_branch():
    cells = json.load(open(NOTEBOOK))["cells"]
    source = "".join("".join(c["source"]) for c in cells)
    assert "BRANCH = 'main'" in source, "DAVIS must stay on main by default"
    assert "AMP = False" in source, "full precision must stay the default"
    assert "git clone --branch {BRANCH}" in source
    assert "git pull origin {BRANCH}" in source
    assert "src/model/resume.py" in source
