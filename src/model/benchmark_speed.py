"""
How much faster would mixed precision make each model? Measure, don't guess.

    python -m src.model.benchmark_speed --dataset kiba --split random
    python -m src.model.benchmark_speed --device cpu --steps 2 --warmup 1   # smoke test

For every model, on real training rows of the chosen split, with the batch
size and optimiser the grid actually uses, this times training steps under
three settings:

    fp32            what every DAVIS cell was trained with
    fp32+benchmark  plus `torch.backends.cudnn.benchmark` (convolution autotuning)
    amp+benchmark   plus mixed precision: float16 autocast and a GradScaler

and reports seconds per batch, the speed-up over fp32, peak GPU memory, how
far float16 moves the model's outputs on the same weights, and whether any
mixed-precision step produced a non-finite gradient. From the measured speed
it projects KIBA's epoch time and hours per cell at 25 epochs (the minimum the
early-stopping rule allows) and 36 (the median the DAVIS histories show).

Nothing here trains a model for real or changes any trainer. Each model is
built from the same classes its trainer uses; only the loop around them is
this file's. It exists so the decision to use mixed precision on KIBA rests on
numbers from the same T4 the grid runs on.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.dataset import BINARY_THRESHOLD

MODELS = ("deepdta", "coldsite_dti", "hyperattentiondti", "moltrans")
# The grid's batch sizes on a >=14 GB GPU (kaggle_binary_grid.ipynb).
BATCH = {"deepdta": 256, "coldsite_dti": 64, "hyperattentiondti": 32, "moltrans": 16}
SETTINGS = ("fp32", "fp32+benchmark", "amp+benchmark")
KIBA_ROWS = {  # (train, valid), results/split_summary.md
    "random": (82778, 11825), "cold_drug": (83807, 12073),
    "cold_target": (85452, 10701), "cold_pair": (58041, 1334),
}
EPOCHS = (25, 36)


# --------------------------------------------------------------------------
# each model, built the way its trainer builds it
# --------------------------------------------------------------------------

def build(model_name: str, csv_dir: str, dataset: str, device: str):
    """(model, loader, optimizer, step_fn, eval_fn) for one model.

    step_fn(batch) returns the loss; eval_fn(batch) the raw outputs. Both run
    inside whatever autocast context the caller sets.
    """
    threshold = BINARY_THRESHOLD[dataset]
    path = os.path.join(csv_dir, "train.csv")
    batch_size = BATCH[model_name]
    bce = nn.BCEWithLogitsLoss()

    if model_name == "deepdta":
        from src.model.deepdta_torch import DeepDTA
        from src.model.train_deepdta import DeepDTADataset

        data = DeepDTADataset(path, "binary", threshold)
        model = DeepDTA().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def forward(batch):
            drug, protein, y = (t.to(device) for t in batch)
            return model(drug, protein), y

        loss = lambda out, y: bce(out, y)  # noqa: E731
        clip = 5.0

    elif model_name == "coldsite_dti":
        from src.model.coldsite_dti import ColdSiteDTI
        from src.model.dataset import load_split

        loader, _v, _t, drug_vocab, protein_vocab = load_split(
            csv_dir, 1000, batch_size, binary_threshold=threshold)
        data = loader.dataset
        model = ColdSiteDTI(len(drug_vocab) + 2, len(protein_vocab) + 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def forward(batch):
            drug, protein, y = (t.to(device) for t in batch)
            pred, _attn = model(drug, protein)
            return pred.squeeze(-1), y.float()

        loss = lambda out, y: bce(out, y)  # noqa: E731
        clip = 5.0

    elif model_name == "hyperattentiondti":
        from src.model.train_hyperattentiondti import HyperAttentionDataset, _import_vendored

        AttentionDTI, hyperparameter, *_ = _import_vendored()
        data = HyperAttentionDataset(path, threshold)
        hp = hyperparameter()
        hp.Batch_size = batch_size
        model = AttentionDTI(hp).to(device)
        weights = [p for n, p in model.named_parameters() if "bias" not in n]
        biases = [p for n, p in model.named_parameters() if "bias" in n]
        opt = torch.optim.AdamW([{"params": weights, "weight_decay": hp.weight_decay},
                                 {"params": biases, "weight_decay": 0}], lr=5e-5)
        ce = nn.CrossEntropyLoss()

        def forward(batch):
            drug, protein, y = (t.to(device) for t in batch)
            return model(drug, protein), y

        loss = lambda out, y: ce(out, y)  # noqa: E731
        clip = None

    elif model_name == "moltrans":
        from src.model.train_moltrans import MolTransDataset, _fit_batch_size, _import_vendored

        BIN_config_DBPE, BIN_Interaction_Flat, drug_enc, protein_enc = _import_vendored()
        data = MolTransDataset(path, threshold, (drug_enc, protein_enc))
        config = BIN_config_DBPE()
        config["batch_size"] = batch_size
        model = BIN_Interaction_Flat(**config).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        def forward(batch):
            d, p, dm, pm, y = (t.to(device) for t in batch)
            _fit_batch_size(model, d.shape[0])
            return model(d, p, dm, pm).squeeze(-1), y

        loss = lambda out, y: bce(out, y)  # noqa: E731
        clip = 5.0
    else:
        raise ValueError(model_name)

    if model_name != "coldsite_dti":
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return model, loader, opt, forward, loss, clip


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _autocast(device, amp):
    if not amp:
        return contextlib.nullcontext()
    # float16 on the GPU is what the T4's tensor cores run; the CPU path is
    # bfloat16 and exists only so the smoke test exercises this branch.
    dtype = torch.float16 if str(device).startswith("cuda") else torch.bfloat16
    return torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                          dtype=dtype)


def time_setting(model_name, setting, csv_dir, dataset, device, steps, warmup, seed=0):
    """Seconds per training batch and per eval batch, plus the checks."""
    amp = setting.startswith("amp")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = "benchmark" in setting
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, loader, opt, forward, loss_fn, clip = build(model_name, csv_dir, dataset, device)
    use_scaler = amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    model.train()
    nonfinite, train_times = 0, []
    for i in range(warmup + steps):
        batch = next_batch()
        _sync(device)
        started = time.perf_counter()
        with _autocast(device, amp):
            out, y = forward(batch)
            loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            nonfinite += 1
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        _sync(device)
        if i >= warmup:
            train_times.append(time.perf_counter() - started)

    # eval speed, and how far float16 moves the outputs on the SAME weights
    model.eval()
    batch = next_batch()
    eval_times = []
    with torch.no_grad():
        for _ in range(3):
            _sync(device)
            started = time.perf_counter()
            with _autocast(device, amp):
                forward(batch)
            _sync(device)
            eval_times.append(time.perf_counter() - started)
        # MolTrans keeps dropout on in eval(); hold the RNG fixed for both passes
        devices = [0] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1)
            reference, _ = forward(batch)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1)
            with _autocast(device, True):
                low, _ = forward(batch)
    diff = (reference.float() - low.float()).abs()
    scale = reference.float().abs().mean().clamp(min=1e-6)

    return {
        "s_per_train_batch": float(np.mean(train_times)),
        "s_per_eval_batch": float(np.median(eval_times)),
        "peak_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3
                    if str(device).startswith("cuda") else None),
        "nonfinite_steps": nonfinite,
        "amp_output_max_abs_diff": float(diff.max()),
        "amp_output_rel_diff": float(diff.mean() / scale),
    }


def project(model_name, s_train, s_eval):
    """KIBA hours per cell at 25 and 36 epochs, per split."""
    batch = BATCH[model_name]
    out = {}
    for split, (train_rows, valid_rows) in KIBA_ROWS.items():
        epoch_s = (train_rows // batch) * s_train + -(-valid_rows // batch) * s_eval
        out[split] = {"epoch_min": epoch_s / 60,
                      **{f"hours_at_{e}": epoch_s * e / 3600 for e in EPOCHS}}
    return out


# --------------------------------------------------------------------------

def sample_split(dataset, split, n_rows, split_root="data/splits", seed=0) -> str:
    """A temporary split dir holding n_rows of the real training rows."""
    src = os.path.join(split_root, dataset, split)
    train = pd.read_csv(os.path.join(src, "train.csv"))
    rows = train.sample(n=min(n_rows, len(train)), random_state=seed)
    tmp = tempfile.mkdtemp(prefix="speed_")
    rows.to_csv(os.path.join(tmp, "train.csv"), index=False)
    rows.head(64).to_csv(os.path.join(tmp, "valid.csv"), index=False)
    rows.head(64).to_csv(os.path.join(tmp, "test.csv"), index=False)
    return tmp


def report(results: dict, meta: dict) -> str:
    lines = [f"# Speed test — {meta['dataset']} {meta['split']}, {meta['device_name']}", "",
             f"torch {meta['torch']}; {meta['steps']} timed steps after {meta['warmup']} warm-up; "
             "grid batch sizes.", "",
             "| model | setting | s/batch (train) | speed-up | peak GB | non-finite steps |",
             "|---|---|---|---|---|---|"]
    for model, per in results.items():
        base = per["fp32"]["s_per_train_batch"]
        for setting in SETTINGS:
            if setting not in per:
                continue
            r = per[setting]
            peak = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "—"
            lines.append(f"| {model} | {setting} | {r['s_per_train_batch']:.3f} | "
                         f"{base / r['s_per_train_batch']:.2f}× | {peak} | {r['nonfinite_steps']} |")
    lines += ["", "## How far float16 moves the outputs (same weights)", "",
              "| model | max abs diff | mean relative diff |", "|---|---|---|"]
    for model, per in results.items():
        r = per.get("amp+benchmark") or per["fp32"]
        lines.append(f"| {model} | {r['amp_output_max_abs_diff']:.4g} | "
                     f"{r['amp_output_rel_diff']:.4g} |")
    lines += ["", "## KIBA hours per cell, projected from the measured speed", "",
              "| model | setting | random: min/epoch | random: h at 25 / 36 ep | "
              "cold_pair: h at 25 / 36 ep |", "|---|---|---|---|---|"]
    for model, per in results.items():
        for setting in ("fp32", "amp+benchmark"):
            if setting not in per:
                continue
            proj = project(model, per[setting]["s_per_train_batch"], per[setting]["s_per_eval_batch"])
            r, cp = proj["random"], proj["cold_pair"]
            lines.append(f"| {model} | {setting} | {r['epoch_min']:.1f} | "
                         f"{r['hours_at_25']:.1f} / {r['hours_at_36']:.1f} | "
                         f"{cp['hours_at_25']:.1f} / {cp['hours_at_36']:.1f} |")
    lines += ["", "A cell over ~10.5 h cannot finish inside one 11-hour Kaggle commit "
              "without resume."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Time fp32 vs mixed precision per model")
    parser.add_argument("--dataset", default="kiba", choices=["davis", "kiba"])
    parser.add_argument("--split", default="random")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--out", help="directory for speed_<dataset>.{json,md}")
    parser.add_argument("--batch", type=int,
                        help="override every batch size -- CPU smoke test only; "
                             "the timings are then not the grid's")
    args = parser.parse_args()
    if args.batch:
        for model in BATCH:
            BATCH[model] = args.batch

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    settings = SETTINGS if args.device.startswith("cuda") else ("fp32", "amp+benchmark")
    need = max(BATCH[m] for m in models) * (args.steps + args.warmup + 2)
    csv_dir = sample_split(args.dataset, args.split, need, args.split_root)
    meta = {"dataset": args.dataset, "split": args.split, "steps": args.steps,
            "warmup": args.warmup, "torch": torch.__version__,
            "device_name": (torch.cuda.get_device_name(0) if args.device.startswith("cuda")
                            else "CPU (smoke test only — speeds are meaningless)")}
    print(f"{meta['device_name']}: {len(models)} models x {len(settings)} settings, "
          f"{args.steps} steps each, on {need:,} real {args.dataset}/{args.split} rows")

    results = {}
    for model in models:
        results[model] = {}
        for setting in settings:
            r = time_setting(model, setting, csv_dir, args.dataset, args.device,
                             args.steps, args.warmup)
            results[model][setting] = r
            print(f"  {model:18s} {setting:15s} {r['s_per_train_batch']:.3f} s/batch  "
                  f"non-finite {r['nonfinite_steps']}", flush=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    text = report(results, meta)
    print("\n" + text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stem = os.path.join(args.out, f"speed_{args.dataset}")
        with open(f"{stem}.json", "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        with open(f"{stem}.md", "w") as f:
            f.write(text)
        print(f"Saved -> {stem}.md")


if __name__ == "__main__":
    main()
