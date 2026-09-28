"""The weekly GPU-hour quota: a ledger that survives sessions, and the rule that pauses training as it fills.

Kaggle gives an account a weekly GPU allowance (30 GPU-hours, stated by the user on 2026-09-25) and refuses
a session once it is spent. A wave that meets that wall mid-epoch loses the epoch and leaves the state in
question, so the harness keeps its own count and stops itself first.

What is counted. Every interval a cell trains on a GPU is written to `<results_root>/quota_ledger.json`
(start, heartbeat every 30 s, end). The ledger travels with the results zip, is merged back on restore, and a
session that died without closing an interval has it closed at its last heartbeat (never later). Usage
outside this harness (the canary, a notebook someone ran by hand) is added with `--add-external`, once per id.
The window is the last 7 days ROLLING, which can only over-count against Kaggle's fixed week: a session from
before Kaggle's reset still counts here, and one inside Kaggle's week always does.

Two units, chosen in `config/harness.json` (`quota_unit`):
  gpu_hours      each GPU worker counts its own wall-clock: a two-GPU hour is two GPU-hours (the conservative
                 reading, the default, and the unit `docs/wave_plan.md` plans in);
  session_hours  wall-clock with every GPU counted once (the union of the intervals).

What it does.
  * usable = weekly quota x (1 - reserve). With 30 h and a 15 % reserve that is 25.5, the figure
    `docs/wave_plan.md` fits the plan against (23.2 mean, 24.7 high case for ACC1/ACC2).
  * Before a cell starts: refused when the usable hours are spent; a FRESH cell is also refused past
    `quota_soft_fraction` of usable (a partial cell may still resume and finish); refused when the hours
    left cannot pay for one epoch (the cell's measured epoch, or `quota_min_start_hours`).
  * While a cell trains, its trainers get `min(session deadline, the moment the usable hours run out)`
    as their stop time, so they stop BETWEEN EPOCHS with the state saved (`src/model/resume.py`) and the
    next session resumes them. The runner also terminates a trainer that overshoots (SIGTERM: finish the
    epoch, save) and kills one only past the nominal weekly quota.

What it cannot do. It cannot see Kaggle's own counter: it counts what this harness ran plus what was declared.

    python -m src.cloud.quota --root results --account ACC1 --status
    python -m src.cloud.quota --root results --account ACC1 --add-external 2.2 --external-id canary-2026-09-29 --note "canary + epoch gate"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass

LEDGER_NAME = "quota_ledger.json"
HEARTBEAT_S = 30.0
UNITS = ("gpu_hours", "session_hours")


@dataclass(frozen=True)
class QuotaCfg:
    weekly_hours: float
    reserve_fraction: float
    soft_fraction: float
    window_days: float
    unit: str
    min_start_hours: float
    workers: int

    @property
    def usable_hours(self) -> float:
        return self.weekly_hours * (1.0 - self.reserve_fraction)

    @property
    def soft_hours(self) -> float:
        return self.usable_hours * self.soft_fraction

    @classmethod
    def from_harness(cls, cfg: dict):
        """None when the harness config declares no weekly quota (the manager is then off)."""
        if "weekly_quota_gpu_hours" not in cfg:
            return None
        q = cls(weekly_hours=float(cfg["weekly_quota_gpu_hours"]),
                reserve_fraction=float(cfg.get("quota_reserve_fraction", 0.15)),
                soft_fraction=float(cfg.get("quota_soft_fraction", 0.95)),
                window_days=float(cfg.get("quota_window_days", 7)),
                unit=str(cfg.get("quota_unit", "gpu_hours")),
                min_start_hours=float(cfg.get("quota_min_start_hours", 0.5)),
                workers=int(cfg.get("required_gpus", 2)))
        if not q.weekly_hours > 0:
            raise ValueError("weekly_quota_gpu_hours must be > 0")
        if not 0 <= q.reserve_fraction < 1:
            raise ValueError("quota_reserve_fraction must be in [0, 1)")
        if not 0 < q.soft_fraction <= 1:
            raise ValueError("quota_soft_fraction must be in (0, 1]")
        if not q.window_days > 0 or q.min_start_hours < 0 or q.workers < 1:
            raise ValueError("quota_window_days must be > 0, quota_min_start_hours >= 0, required_gpus >= 1")
        if q.unit not in UNITS:
            raise ValueError(f"quota_unit must be one of {UNITS}, got {q.unit!r}")
        return q


class Ledger:
    """The account's GPU intervals. Thread-safe; every change is written atomically."""

    def __init__(self, path: str, account: str, clock=time.time):
        self.path, self.account, self.clock = path, account, clock
        self.entries: list = []
        self.closed_after_crash: list = []
        self._lock = threading.RLock()
        self._load()

    # -- persistence --------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as handle:
            data = json.load(handle)
        if data.get("account") not in (None, self.account):
            raise ValueError(f"{self.path} belongs to account {data.get('account')!r}, not {self.account!r}: "
                             f"each account has its own quota")
        self.entries = list(data.get("entries", []))
        for entry in self.entries:
            if entry["kind"] == "train" and entry.get("end") is None:
                entry["end"] = max(entry.get("heartbeat") or entry["start"], entry["start"])
                entry["closed_by"] = "last heartbeat (the session that opened it did not close it)"
                self.closed_after_crash.append(entry["id"])
        if self.closed_after_crash:
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as handle:
            json.dump({"account": self.account, "schema": 1, "entries": self.entries}, handle, indent=1)
        os.replace(tmp, self.path)

    # -- writing ------------------------------------------------------------------------
    def open(self, cell_id: str, gpu: int) -> str:
        with self._lock:
            now = self.clock()
            eid = f"{cell_id}@gpu{gpu}@{now:.3f}"
            self.entries.append({"id": eid, "kind": "train", "cell": cell_id, "gpu": gpu,
                                 "start": now, "end": None, "heartbeat": now})
            self._save()
            return eid

    def _find(self, eid: str) -> dict:
        for entry in self.entries:
            if entry["id"] == eid:
                return entry
        raise KeyError(eid)

    def beat(self, eid: str) -> None:
        with self._lock:
            self._find(eid)["heartbeat"] = self.clock()
            self._save()

    def close(self, eid: str) -> None:
        with self._lock:
            entry = self._find(eid)
            if entry.get("end") is None:
                entry["end"] = self.clock()
                entry["heartbeat"] = entry["end"]
                self._save()

    def add_external(self, ext_id: str, hours: float, note: str = "") -> bool:
        """Usage outside this harness, in the ledger's unit. Idempotent per id. False if already there."""
        if not hours >= 0:
            raise ValueError("hours must be >= 0")
        with self._lock:
            if any(e["id"] == ext_id for e in self.entries):
                return False
            self.entries.append({"id": ext_id, "kind": "external", "hours": float(hours),
                                 "at": self.clock(), "note": note})
            self._save()
            return True

    def merge_file(self, path: str) -> int:
        """Add the entries of another ledger (a previous session's output). Returns how many were new."""
        with open(path) as handle:
            data = json.load(handle)
        if data.get("account") not in (None, self.account):
            raise ValueError(f"{path} belongs to account {data.get('account')!r}, not {self.account!r}")
        with self._lock:
            known = {e["id"] for e in self.entries}
            added = 0
            for entry in data.get("entries", []):
                if entry["id"] in known:
                    continue
                entry = dict(entry)
                if entry["kind"] == "train" and entry.get("end") is None:
                    entry["end"] = max(entry.get("heartbeat") or entry["start"], entry["start"])
                    entry["closed_by"] = "last heartbeat (merged from a ledger that never closed it)"
                self.entries.append(entry)
                added += 1
            if added:
                self._save()
            return added

    # -- reading ------------------------------------------------------------------------
    def used_hours(self, unit: str, window_days: float, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        cutoff = now - window_days * 86400.0
        spans, external = [], 0.0
        with self._lock:
            for entry in self.entries:
                if entry["kind"] == "external":
                    if entry["at"] >= cutoff:
                        external += float(entry["hours"])
                    continue
                start = max(entry["start"], cutoff)
                end = min(entry["end"] if entry.get("end") is not None else now, now)
                if end > start:
                    spans.append((start, end))
        if unit == "gpu_hours":
            return sum(e - s for s, e in spans) / 3600.0 + external
        merged, total = [], 0.0
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        total = sum(e - s for s, e in merged)
        return total / 3600.0 + external


class Quota:
    def __init__(self, cfg: QuotaCfg, ledger: Ledger, clock=time.time):
        self.cfg, self.ledger, self.clock = cfg, ledger, clock

    @classmethod
    def from_harness(cls, harness_cfg: dict, results_root: str, account: str, clock=time.time):
        qcfg = QuotaCfg.from_harness(harness_cfg)
        if qcfg is None:
            return None
        return cls(qcfg, Ledger(os.path.join(results_root, LEDGER_NAME), account, clock), clock)

    def used(self, now: float | None = None) -> float:
        return self.ledger.used_hours(self.cfg.unit, self.cfg.window_days, now)

    def remaining(self, now: float | None = None) -> float:
        """Usable hours left (negative once over)."""
        return self.cfg.usable_hours - self.used(now)

    def exhausted(self, now: float | None = None) -> bool:
        return self.used(now) >= self.cfg.usable_hours

    def over_nominal(self, now: float | None = None) -> bool:
        return self.used(now) >= self.cfg.weekly_hours

    def stop_at(self, now: float | None = None) -> float:
        """Unix time at which the usable hours run out if every GPU worker keeps training. Both
        workers are assumed busy (the pessimistic burn rate), so a trainer never plans past it."""
        now = self.clock() if now is None else now
        burn = self.cfg.workers if self.cfg.unit == "gpu_hours" else 1
        return now + max(self.remaining(now), 0.0) / burn * 3600.0

    def can_start(self, epoch_seconds: float, resume: bool, now: float | None = None) -> tuple:
        c = self.cfg
        used = self.used(now)
        if used >= c.usable_hours:
            return False, (f"weekly quota spent: {used:.2f} of {c.usable_hours:.2f} usable {c.unit.replace('_', '-')} "
                           f"({c.weekly_hours:g} x (1 - {c.reserve_fraction:g}))")
        if not resume and used >= c.soft_hours:
            return False, (f"approaching the weekly quota: {used:.2f} of {c.usable_hours:.2f} usable hours used "
                           f"(no fresh cell starts past {c.soft_fraction:.0%}); a partial cell may still resume")
        remaining = c.usable_hours - used
        need = max(epoch_seconds / 3600.0, c.min_start_hours)
        if remaining < need:
            return False, f"only {remaining:.2f} usable hours left; one epoch needs about {need:.2f}"
        return True, f"quota ok: {used:.2f} of {c.usable_hours:.2f} usable hours used"

    def report(self, now: float | None = None) -> dict:
        c = self.cfg
        return {"unit": c.unit, "weekly": c.weekly_hours, "reserve_fraction": c.reserve_fraction,
                "usable": round(c.usable_hours, 3), "window_days": c.window_days,
                "used": round(self.used(now), 3), "remaining_usable": round(self.remaining(now), 3),
                "closed_after_crash": list(self.ledger.closed_after_crash)}


def restore_ledger(account: str, results_root: str, input_root: str) -> dict:
    """Merge every `quota_ledger.json` found under `input_root` (the previous versions' outputs) into this
    session's ledger. Same-id entries are not duplicated; a ledger of another account is an error."""
    report = {"merged_files": 0, "new_entries": 0, "problems": []}
    if not os.path.isdir(input_root):
        return report
    ledger = Ledger(os.path.join(results_root, LEDGER_NAME), account)
    for folder, _dirs, names in os.walk(input_root):
        for name in names:
            path = os.path.join(folder, name)
            if name != LEDGER_NAME or os.path.abspath(path) == os.path.abspath(ledger.path):
                continue
            try:
                report["new_entries"] += ledger.merge_file(path)
                report["merged_files"] += 1
            except (ValueError, KeyError, OSError) as exc:
                report["problems"].append(f"{path}: {exc}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results", help="the results root that holds quota_ledger.json")
    ap.add_argument("--account", required=True)
    ap.add_argument("--harness", default=os.path.join("config", "harness.json"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--add-external", type=float, metavar="HOURS",
                    help="usage outside this harness, in the ledger's unit (e.g. the canary)")
    ap.add_argument("--external-id", help="a stable id, so adding the same usage twice is a no-op")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    with open(args.harness) as handle:
        cfg = json.load(handle)
    quota = Quota.from_harness(cfg, args.root, args.account)
    if quota is None:
        print("no weekly quota configured in", args.harness)
        return 2
    if args.add_external is not None:
        if not args.external_id:
            ap.error("--add-external needs --external-id")
        added = quota.ledger.add_external(args.external_id, args.add_external, args.note)
        print(f"{'added' if added else 'already recorded (no change)'}: {args.external_id} = {args.add_external:g} h")
    print(json.dumps(quota.report(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
