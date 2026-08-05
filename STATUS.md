# Part 1 status

Last updated: 2026-08-04. Regenerate the numbers with `python -m pytest tests/ -q`
and `python -m src.data.ground_truth`.

This file tracks Part 1 against the checklists in each guide. **Part 1 is not
complete.** Five items remain, all of them blocked on data or compute rather
than on code. They are listed at the bottom with the exact command to run.

---

## Track A — 124AD0008 (Data, splits, baselines)

| | Item | Status |
|---|---|---|
| A1 | DAVIS + KIBA load with stats | ✅ `src/data/load_data.py` |
| A2 | Four splits, train/valid/test, both datasets | ✅ `src/data/build_splits.py` |
| A3 | Leakage check in code | ✅ + 18 tests, incl. injected-leakage cases |
| A4 | Split files saved | ⏳ code ready, needs the DeepDTA data files |
| A5 | Split summary table | ✅ auto-written to `results/split_summary.md` |
| A6 | Antiviral subset, 5 targets | ⏳ **1 of 5 in the committed file**; rebuilder ready |
| A7 | Binding-site ground truth | ⚠️ usable but needs re-fetch (see below) |
| A8 | Ground-truth README | ✅ `data/GROUND_TRUTH_README.md` |
| A9 | Three baselines runnable | ⏳ all three vendored, only MolTrans run |
| A10 | Results table, 3×4×2 | ❌ **1 of 24 cells** |

## Track B — 124AD0015 (Core model)

| | Item | Status |
|---|---|---|
| B1 | Drug encoder | ✅ tested, padding-invariant |
| B2 | Protein encoder, attention exposed | ✅ tested, zero weight on padding |
| B3 | Fusion + prediction head | ✅ end-to-end on dummy data |
| B4 | Training loop with checkpointing | ✅ verified `--dummy` |
| B5 | Trained on real DAVIS/KIBA, all four splits | ❌ **blocked on A4** |
| B6 | SMILES vocab from train split only | ✅ tested |

## Track C — 124AD0067 (Evaluation, case study, literature)

| | Item | Status |
|---|---|---|
| C1 | `precision@k` module | ✅ rewritten with input guards |
| C2 | Multiple k values | ✅ + achievable-ceiling reporting |
| C3 | Tested on several cases incl. edge cases | ✅ **171 tests**, was 0 |
| C4 | Permutation significance test | ✅ + split-level test |
| C5 | Headline figure mock-up | ✅ `src/evaluation/plots.py` |
| C6 | Antiviral reference sheet | ✅ `antiviral_targets_reference.md` |
| C7 | Five-paper differentiation | ✅ `literature_differentiation.md` |
| C8 | Cross-check the 5 targets against A's data | ✅ done — it found A6 |

---

## What changed in this pass

### Correctness fixes

**The Track A → Track C coordinate mismatch.** UniProt reports 1-indexed
inclusive ranges; `precision_at_k` expects 0-indexed positions. Nothing
converted between them, and the error is silent — a model with *perfect*
attention scored 0.67 instead of 1.00. `src/data/ground_truth.py` now owns
that conversion, and `test_off_by_one_regression` pins it.

**`precision_at_k` returned wrong numbers instead of raising.** `k=500` on a
300-residue protein returned 0.012: `argsort(x)[-500:]` yields 300 indices,
divided by 500 anyway. `k=0` raised `ZeroDivisionError`. Both now raise with a
message naming the problem.

**Tie-breaking bias.** `np.argsort` is stable, so `[-k:]` on tied attention
returned the highest indices — biasing selection toward the C-terminus. On a
flat attention map, sites near the protein's end scored 1.0 on a model that had
learned nothing. Ties are now broken at random against a seeded generator.

**Unsafe shuffle in the split builder.** `rng.shuffle` on a pandas
`StringArray` warns that it may leave duplicates. A duplicated ID puts the same
drug in train and test and voids the paper's central claim. Now shuffles a
`dtype=object` array.

**`explain()` length mismatch.** It counted non-pad tokens while the encoder
measured to the last non-pad position. These disagree on any sequence with an
interior pad, returning a shorter array than the residues it covers and
misaligning every ground-truth index past that point.

**Ground-truth contamination.** The fetcher collected UniProt's `Site`
catch-all, which carries protease cleavage points and chromosomal breakpoints —
positions no drug binds to, scored as correct answers. ~136 DAVIS and ~68 KIBA
annotations. Removed from the fetcher; filtered by description heuristic in the
adapter until the files are re-fetched.

**`coverage_report` hid its own losses.** It summed drop counters over the
*filtered* mapping, so a protein whose sites were all discarded took its counts
out of the total. Truncation losses read as 22 when the real figure was 283.

**p-value floor.** The permutation test used a bare mean, which can return
exactly 0.0 — infinite confidence from 1000 draws. Now `(1+hits)/(1+trials)`.

**Statistical framing.** Added a split-level permutation test. Per-protein
p-values across 400 proteins are 400 hypothesis tests; ~20 land under 0.05 by
luck. The paper needs one test on the mean, per split.

### Added

- `src/data/ground_truth.py` — the A→C adapter, with explicit truncation policy
- `tests/` — 160 tests across 7 files, including an end-to-end integration test
  that runs model → `explain()` → adapter → precision@k → p-value
- `src/data/extract_antiviral.py` — replaces the single-target pipeline; refuses
  to write unless all five targets are present; converts affinities to p-scale
  and keeps the measurement type instead of pooling IC50/Ki/Kd/EC50
- `achievable_ceiling` / `normalised_precision_at_k` — six sites at k=20 caps at
  0.30, so raw precision is not comparable across proteins
- `summarise_splits` — surfaces that cold-pair trains on ~54% of the pairs the
  other levels get, a confound that would otherwise be read as pure difficulty
- `src/evaluation/run_ladder.py` — the full ladder: loads checkpoints, collects
  explanations, computes fidelity + significance at every level, and writes the
  JSON, the markdown table and the headline figure. Runs today with `--dummy`,
  runs unchanged on real checkpoints in October. Guide C Step 3 asks for exactly
  this: build the plumbing before the numbers arrive, not under time pressure.
- `.github/workflows/tests.yml` — CI runs the suite, verifies the ground truth
  still converts, and smoke-tests the ladder on every push
- `conftest.py` + `pytest.ini` — the suite runs identically from any directory

### Consolidated / removed

- Three overlapping fetchers (`binding_sites.py`, `fetch_binding_sites.py`,
  `fetch_kiba_binding_sites.py`) that filtered features differently → one
- `clean_antiviral.py` → `src/data/extract_antiviral.py`

### A sanity floor worth keeping

`test_an_untrained_model_scores_around_chance_not_above_it` runs an untrained
model through the whole pipeline and asserts the result is **not** significant.
If it ever fires, the metric is measuring an artefact — padding, index bias,
tie ordering — and every real number produced afterwards is suspect.

---

## What is left, and why

None of these are code problems. All five need data or compute this environment
did not have (`rest.uniprot.org` and `bindingdb.org` are unreachable here, and
the DeepDTA data files are not in the repo).

| # | Item | Owner | Command | Est. |
|---|---|---|---|---|
| 1 | Rebuild the antiviral subset | A | download `BindingDB_All.tsv`, then `python -m src.data.extract_antiviral --source data/raw/BindingDB_All.tsv` | ~1h, mostly download |
| 2 | Re-fetch ground truth with feature types | A | `python -m src.data.fetch_binding_sites --dataset davis` (and `kiba`) | ~30 min |
| 3 | Resolve the 33 unmapped DAVIS targets | A | manual gene→UniProt lookup; check `*_provenance.json` for `not_found` | ~2h |
| 4 | Train ColdSite-DTI on 4 splits × 2 datasets | B | `python -m src.model.train --split-dir data/splits/davis/cold_target --dataset davis --split cold_target` | 8 runs, HPC |
| 5 | Fill the remaining 23 baseline cells | A | per `src/data/baselines/README.md` | the long pole |

Do them in that order. 1 and 2 unblock 4, and 4 unblocks the first real
precision@k number.

After 1 and 2, delete `test_committed_antiviral_file_is_still_incomplete` in
`tests/test_antiviral.py` — it is a deliberate guard on the current broken
artefact and should fail once the artefact is fixed.

## Running the checks

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 171 tests, ~5s
python -m src.data.ground_truth     # ground-truth coverage report
```

## Producing the headline figure

Once items 1, 2 and 4 above are done, the ladder is one command:

```bash
python -m src.evaluation.run_ladder \
    --dataset davis \
    --ground-truth data/davis_ground_truth_sites.json \
    --checkpoint-dir results \
    --accuracy-json results/accuracy_by_level.json
```

It writes `results/ladder_davis.json`, `results/ladder_davis.md` (the table for
the paper) and `results/headline_davis.png`.

Two behaviours are deliberate. It **refuses to draw the figure** without
accuracy values, because fidelity plotted alone is half the paper's claim. And
`--dummy` output is tagged `DUMMY_PLACEHOLDER` in every filename so a synthetic
run cannot later be mistaken for a result.
