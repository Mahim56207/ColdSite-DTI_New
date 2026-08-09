# Project status

> **Framing changed.** The project is now an audit of published DTI
> interpretability claims, not a single-model paper. Read
> `docs/00_MASTER_PLAN_V2.md`, then your `docs/PART2_GUIDE_<roll>.md`.

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
| C3 | Tested on several cases incl. edge cases | ✅ **271 tests**, was 0 |
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
- `summarise_splits` — surfaces the cold-pair volume confound, which would
  otherwise be read as pure difficulty. **Two different numbers, do not mix
  them up:** cold-pair *trains* on roughly **71%** of the pairs the other levels
  get, and *uses* roughly **54%** of all measured pairs once the rows it
  discards (one cold entity, not two) are counted. The `pct_of_largest_split`
  column reports the second. Quoting 54% as the training ratio overstates the
  confound by about a factor of two.
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
| 4 | Train ColdSite-DTI: 2 datasets × 4 splits × 3 seeds | B | `python -m src.model.run_grid --preflight`, then `python -m src.model.run_grid` | **24 runs**, HPC |
| 5 | Fill the remaining 23 baseline cells | A | per `src/data/baselines/README.md` | the long pole |

Do them in that order. Item 4 is blocked on the **split files**, not on 1–3:
`data/splits/` is empty and the DeepDTA files `build_splits.py` reads are not
in the repo. Items 1 and 2 gate the *metric* rather than the training, and 4
unblocks the first real precision@k number either way.

After 1 and 2, delete `test_committed_antiviral_file_is_still_incomplete` in
`tests/test_antiviral.py` — it is a deliberate guard on the current broken
artefact and should fail once the artefact is fixed.

## Running the checks

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 387 tests, ~6s
python -m src.data.ground_truth     # ground-truth coverage report
```

## Producing the headline figure

Once items 1, 2 and 4 above are done, the ladder is one command per seed. Run
`run_faithfulness` first — it produces the `--accuracy-json` file the ladder
needs (next section).

```bash
python -m src.evaluation.run_ladder \
    --dataset davis --seed 1 \
    --ground-truth data/davis_ground_truth_sites.json \
    --checkpoint-dir results \
    --accuracy-json results/accuracy_davis_seed1.json
```

It writes `results/ladder_davis_seed1.json`, `results/ladder_davis_seed1.md`
(the table for the paper) and `results/headline_davis_seed1.png`.

`--seed` selects which training run to read, and it appears in the output names
too: three ladder runs writing one `ladder_davis.json` would overwrite each
other exactly as unseeded checkpoints did. Checkpoint and result filenames are
built by `src/model/checkpoint_naming.py` at both ends — the writer
(`src.model.train`) and the reader (`run_ladder`) import the same function, so
the two cannot drift apart:

    results/coldsite_dti_{dataset}_{split}_{task}_seed{N}.pt
    results/{dataset}_{split}_{task}_seed{N}_results.json

Two behaviours of `run_ladder` are deliberate. It **refuses to draw the figure**
without accuracy values, because fidelity plotted alone is half the paper's
claim. And `--dummy` output is tagged `DUMMY_PLACEHOLDER` in every filename so a
synthetic run cannot later be mistaken for a result.

## Faithfulness and the accuracy hand-off

```bash
python -m src.evaluation.run_faithfulness --dummy          # no data needed

python -m src.evaluation.run_faithfulness \
    --dataset davis --seed 1 --checkpoint-dir results
```

It writes `results/faithfulness_davis_seed1.json`,
`results/faithfulness_davis_seed1.md` and
`results/faithfulness_davis_seed1.png` — comprehensiveness against its
random-masking control per level, with the **delta** as the only column that is
a result — plus `results/accuracy_davis_seed1.json`, the flat
`{level: accuracy}` file `run_ladder --accuracy-json` reads. Accuracy is read
out of the trainer's own `*_results.json` rather than recomputed, so the
figure's accuracy axis cannot disagree with the runs it reports.


---

# Audit reframe — what was added

New modules, all tested, all runnable today:

| module | purpose |
|---|---|
| `src/evaluation/faithfulness.py` | comprehensiveness, sufficiency, AOPC — **with random-masking controls**. Promoted from optional stretch goal to core. |
| `src/evaluation/model_registry.py` | the adapter contract every audited model must satisfy, plus `validate_adapter` and a uniform-attention control |
| `src/evaluation/aggregate.py` | seed aggregation (mean ± std, flags <3 seeds), Holm-Bonferroni across the grid, the audit table |
| `src/evaluation/target_family.py` | kinase / non-kinase stratification — the confound control |
| `src/evaluation/run_audit.py` | the audit grid: models × splits × datasets × seeds, with Holm correction applied once over the whole family |
| `src/evaluation/baseline_adapters.py` | registered stubs for DeepDTA / HyperAttentionDTI / MolTrans that raise with instructions until filled in |
| `src/evaluation/plots.py` | rewritten: multi-model curves with seed error bars, control floor, stratified panels, faithfulness-vs-control bars |

## The confound, measured

`python -m src.evaluation.target_family` on the committed data:

| | DAVIS | KIBA |
|---|---|---|
| targets | 409 | 224 |
| kinase | 230 | 0 |
| **non-kinase** | **0** | **0** |
| control usable | **no** | **no** |

KIBA is zero because it uses UniProt accessions, not gene symbols, and cannot be
classified without a mapping. **The control arm does not currently exist.** This
is the highest-priority gap in the project.

## Remaining work, by owner

| # | Item | Owner | Blocking |
|---|---|---|---|
| 1 | Antiviral rebuild — 5 targets, ≥20 non-kinase | A | the entire control arm |
| 2 | Ground-truth re-fetch with feature types | A | metric validity |
| 3 | KIBA accession → gene mapping | A | control on KIBA |
| 4 | 33 unmapped DAVIS targets | A | coverage |
| 5 | **Split files for both datasets** | A | items 6, 7 below — the whole grid |
| 6 | 24 training runs (2 × 4 × 3 seeds) | B | every real number |
| 7 | Baseline adapters (3 models) | A + B | the audit framing |
| 8 | Differentiation doc rewrite | C | Related Work |
| 9 | Audit grid + Holm correction | C | Results |

Items 1–3 unblock 6, which unblocks 9. **Item 5 is the hard blocker for Track
B**: `data/splits/` is empty and the DeepDTA source files under
`src/data/baselines/deepdta/data/` that `build_splits.py` reads are not present.

---

# Track B (124AD0015) Part 2 — state

| Item | Status |
|---|---|
| Seed-aware checkpoint naming, shared writer/reader | ✅ `src/model/checkpoint_naming.py` |
| Faithfulness runner + random-masking control + delta | ✅ `src/evaluation/run_faithfulness.py` |
| Accuracy hand-off to Track C | ✅ `accuracy_{dataset}_seed{N}.json` |
| Truncation decision (`exclude`, max_len 1000) | ✅ decided, evidenced, written up |
| Cold-pair volume decision (report, don't subsample) | ✅ decided; `--train-subsample` ready for the control |
| 24-cell grid runner with preflight + one-cell validation | ✅ `src/model/run_grid.py` |
| Attention extraction for HyperAttentionDTI / MolTrans | ✅ `src/evaluation/attention_projection.py` |
| `validate_adapter` passing for ColdSite-DTI | ✅ padded and unpadded |
| Methods / Model Architecture draft | ✅ `paper/methods_track_b.md` |
| **24 training runs** | ⛔ blocked on item 5 |
| **Faithfulness on real checkpoints** | ⛔ blocked on the grid |
| **Volume-matched sensitivity run** | ⛔ blocked on the grid |

Everything above the line runs today and is covered by the test suite. The
three blocked items are blocked on data, not on code:

```bash
python -m src.model.run_grid --preflight   # says exactly what is missing
```

## Grid semantics, settled

**2 datasets × 4 splits × 3 TRAINING seeds = 24 runs**, not 72. The three seeds
vary weight initialisation and batch order on one fixed split per cell. The
repository is consistent on this: the Part 2 guide's loop varies only `--seed`
against a seed-independent `--split-dir`; `build_all_splits()` takes no seed
argument and writes one split per cell; `run_audit.build_grid` has a single seed
axis; and both the guide and this file state the count as 24.

One consequence belongs in the paper: seed error bars measure **initialisation
variance, not split-selection variance**.
