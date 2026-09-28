"""Collect the numbers the manuscript cites from the certification outputs into one tidy CSV.

Reads results/certification/step*.txt (written by the step scripts beside this file) and writes
results/certification/key_numbers.csv with columns name,value,low,high,p,source. Nothing is computed
here: every value is copied from the output line named in `source`, so the manuscript can cite the CSV
with the same rounding-tolerant tags it uses for every other table.

    python scripts/certification/key_numbers.py
"""
import csv, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "results", "certification")
def lines(name): return open(os.path.join(D, name), encoding="utf-8").read().split("\n")
rows = []
def add(name, value, source, low="", high="", p=""): rows.append(dict(name=name, value=value, low=low, high=high, p=p, source=source))
def grab(pattern, name, ln_list, i):
    for n, l in enumerate(ln_list, 1):
        m = re.search(pattern, l)
        if m: return m, n
    raise SystemExit(f"pattern for {name} not found in {i}")
f = "step1_disagreement.txt"; L = lines(f)
m, n = grab(r"CELLS: (\d+)\s+\(DAVIS (\d+), KIBA (\d+)\)", "cells", L, f); add("cells_total", m[1], f"{f}:{n}"); add("cells_davis", m[2], f"{f}:{n}"); add("cells_kiba", m[3], f"{f}:{n}")
m, n = grab(r"SEEDS DISAGREE.*?:\s+(\d+) of (\d+)", "dis", L, f); add("seeds_disagree_exact", m[1], f"{f}:{n}")
m, n = grab(r"ALL THREE pass: (\d+).*NONE pass: (\d+)", "a3", L, f); add("all_three_pass", m[1], f"{f}:{n}"); add("none_pass", m[2], f"{f}:{n}")
m, n = grab(r"exact-chance (\d+) of", "spread", L, f); add("spread_exceeds_distance", m[1], f"{f}:{n}")
m, n = grab(r"DAVIS cells that disagree: (\d+)\s+KIBA: (\d+)", "dk", L, f); add("disagree_davis", m[1], f"{f}:{n}"); add("disagree_kiba", m[2], f"{f}:{n}")
m, n = grab(r"Friedman\s*: cells with p<0.05 \(uncorrected\) = (\d+) of \d+;\s+after Holm over \d+ = (\d+)", "fr", L, f); add("friedman_uncorrected", m[1], f"{f}:{n}"); add("friedman_holm", m[2], f"{f}:{n}")
m, n = grab(r"permutation : cells with p<0.05 \(uncorrected\) = (\d+) of \d+;\s+after Holm over \d+ = (\d+)", "pe", L, f); add("permutation_uncorrected", m[1], f"{f}:{n}"); add("permutation_holm", m[2], f"{f}:{n}")
f = "step1b_exact_p_validation.txt"; L = lines(f)
m, n = grab(r"exact convolution p = ([\d.]+) .* repo's 1,000-perm p = ([\d.]+)", "ex", L, f); add("p_exact_moltrans_random_seed1", m[1], f"{f}:{n}"); add("p_sampled_moltrans_random_seed1", m[2], f"{f}:{n}")
for a in ("0.010", "0.100"):
    m, n = grab(rf"^\s+{a}\s+(\d+)\s+\d+\s+\d+", a, L, f); add(f"disagree_at_alpha_{a}", m[1], f"{f}:{n}", low=a)
f = "step1c_rerun_noise.txt"; L = lines(f)
m, n = grab(r"re-run noise sd .* = ([\d.]+) ; between-seed sd .* = ([\d.]+) ; ratio = ([\d.]+)", "noise", L, f); add("rerun_noise_sd", m[1], f"{f}:{n}"); add("between_seed_sd", m[2], f"{f}:{n}"); add("noise_ratio", m[3], f"{f}:{n}")
f = "step3_leak.txt"; L = lines(f); level = None
for n, l in enumerate(L, 1):
    if "[cold_target]" in l: level = "ct"
    if "[cold_pair]" in l: level = "cp"
    m = re.search(r"leak \(seqmatched - seqclean\) on (\w+)\s+rows: .*mean ([+-][\d.]+)\s+sd [\d.]+\s+95% t-CI \[([+-][\d.]+), ([+-][\d.]+)\]\s+one-sample t p = ([\d.]+)", l)
    if m and level: add(f"leak_{level}_{m[1]}_rows", m[2].lstrip("+"), f"{f}:{n}", m[3].lstrip("+"), m[4].lstrip("+"), m[5])
f = "step4_holm.txt"; L = lines(f)
for ds in ("davis", "kiba"): pass
k = 0
for n, l in enumerate(L, 1):
    m = re.search(r"per-seed tests \(Holm over (\d+) seed-level exact p\): (\d+) seed-runs survive", l)
    if m:
        ds = "davis" if k == 0 else "kiba"; k += 1
        add(f"holm_per_seed_{ds}_tests", m[1], f"{f}:{n}"); add(f"holm_per_seed_{ds}_survivors", m[2], f"{f}:{n}")
with open(os.path.join(D, "key_numbers.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "value", "low", "high", "p", "source"]); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> results/certification/key_numbers.csv")
