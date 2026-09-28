"""Harness configuration and the wave manifest: who trains which cell, and the numbers the
pre-flight and self-stop read. Nothing here is a guess about Kaggle's quotas; the two time
figures are declared in `config/harness.json` with their source, and so is the weekly GPU quota the
user stated on 2026-09-25 (30 GPU-hours per account), which `src/cloud/quota.py` enforces."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

HARNESS_PATH = os.path.join("config", "harness.json")
CLOUD_MANIFEST_PATH = os.path.join("config", "cloud_manifest.json")
WAVES_PATH = os.path.join("config", "waves.json")

MODELS = ("deepdta", "coldsite_dti", "hyperattentiondti", "moltrans", "drugban")
DATASETS = ("davis", "kiba")
LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


@dataclass(frozen=True)
class Cell:
    dataset: str
    model: str
    level: str
    seed: int

    @property
    def id(self) -> str:
        # the spelling src/evaluation/accuracy_table.py already uses
        return f"{self.dataset}_{self.level}_{self.model}_seed{self.seed}"

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        cell = cls(str(d["dataset"]), str(d["model"]), str(d["level"]), int(d["seed"]))
        if cell.dataset not in DATASETS or cell.model not in MODELS or cell.level not in LEVELS:
            raise ValueError(f"unknown cell {d!r}")
        if cell.seed < 1:
            raise ValueError(f"seed must be >= 1: {d!r}")
        return cell


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def load_harness(path: str = HARNESS_PATH) -> dict:
    cfg = load_json(path)
    for key in ("session_limit_hours", "safety_margin_minutes", "required_gpus",
                "gpu_name_contains", "disk_reserve_gb", "checkpoint_bytes"):
        if key not in cfg:
            raise ValueError(f"{path} is missing {key!r}")
    if not (cfg["session_limit_hours"] > 0 and cfg["safety_margin_minutes"] >= 0):
        raise ValueError("session_limit_hours must be > 0 and safety_margin_minutes >= 0")
    if cfg["safety_margin_minutes"] >= cfg["session_limit_hours"] * 60:
        raise ValueError("the safety margin leaves no time to train")
    missing = [m for m in MODELS if m not in cfg["checkpoint_bytes"]]
    if missing:
        raise ValueError(f"checkpoint_bytes has no entry for {missing}")
    from src.cloud.quota import QuotaCfg
    QuotaCfg.from_harness(cfg)            # validates the quota keys when they are present
    return cfg


def stop_at(session_start: float, cfg: dict) -> float:
    """Unix time at which training must have stopped: limit minus margin."""
    return session_start + cfg["session_limit_hours"] * 3600 - cfg["safety_margin_minutes"] * 60


def load_waves(path: str = WAVES_PATH) -> dict:
    """{'wave': ..., 'accounts': {name: {'gpu0': [Cell], 'gpu1': [Cell]}}}, validated:
    every cell listed at most once across the whole manifest."""
    raw = load_json(path)
    accounts, seen = {}, {}
    for name, queues in raw["accounts"].items():
        accounts[name] = {}
        for gpu in ("gpu0", "gpu1"):
            cells = [Cell.from_dict(c) for c in queues.get(gpu, [])]
            for cell in cells:
                if cell.id in seen:
                    raise ValueError(f"{cell.id} is planned for both {seen[cell.id]} and "
                                     f"{name}/{gpu}: a cell must be trained once")
                seen[cell.id] = f"{name}/{gpu}"
            accounts[name][gpu] = cells
    return {"wave": raw.get("wave"), "schema": raw.get("schema"), "accounts": accounts}
