# Project status

> **Framing changed.** The project is now an audit of published DTI
> interpretability claims, not a single-model paper. Read
> `docs/00_MASTER_PLAN_V2.md`, then your `docs/PART2_GUIDE_<roll>.md`.

Last updated: 2026-08-09. Regenerate the numbers with `python -m pytest tests/ -q`
and `python -m src.data.ground_truth`.

This file tracks Part 1 against the checklists in each guide. **Part 1 is not
complete.** The remaining items are listed at the bottom with the exact command
to run. Track A's step-by-step is `docs/RUNBOOK_124AD0008.md`.

> **2026-08-09 — the splits exist. Track B is unblocked.**
> `python -m src.data.build_splits` has been run for real against the canonical
> DeepDTA-direct loader. All 8 split directories are built (2 datasets × 4
> levels, three-way, leakage checks clean). `data/splits/` is gitignored, so
> regenerate locally: put the DeepDTA files at
> `src/data/baselines/deepdta/data/{davis,kiba}/` then run the command.
> Counts are in `results/split_summary.md`.
>
> Note for 124AD0015: **DAVIS cold-pair validation is 264 rows.** Inherent to
> requiring both drug and target unseen, not a bug — but early stopping on it
> will be noisy, so decide how to handle that before the grid runs.

---

## Track A — 124AD0008 (Data, splits, baselines)

| | Item | Status |
|---|---|---|
| A1 | DAVIS + KIBA load with stats | ✅ `src/data/load_data.py` — 30,056/68/442 and 118,254/2,111/229, matching the published benchmark |
| A2 | Four splits, train/valid/test, both datasets | ✅ `src/data/build_splits.py` |
| A3 | Leakage check in code | ✅ + 18 tests, incl. injected-leakage cases |
| A4 | Split files saved | ✅ **run for real 2026-08-09**, all 8 dirs, leakage clean |
| A5 | Split summary table | ✅ auto-written to `results/split_summary.md` |
| A6 | Antiviral subset | ✅ **10,548 pairs, 3 targets** (HIV-1 protease, HIV-1 RT, influenza NA). SARS-CoV-2 unavailable — see below |
| A7 | Binding-site ground truth | ✅ **re-fetched** — DAVIS 442 targets / 406 usable, KIBA 229 / 212, `type` on every feature, contamination gone |
| A8 | Ground-truth README | ✅ `data/GROUND_TRUTH_README.md` |
| A9 | Three baselines runnable | ⏳ all three vendored, only MolTrans run |
| A10 | Results table, 3×4×2 | ❌ **1 of 24 cells** — now the main remaining Track A item |
| A11 | KIBA accession → gene map | ✅ `data/kiba_uniprot_to_gene.json`, 229/229 with a gene symbol |
| A12 | Unmapped DAVIS targets | ✅ resolved or documented — 19 unresolvable (real UniProt annotation gaps), 3 fixed by hand |
| A13 | **Non-kinase control panel** | ✅ **60 distinct human targets, 21,145 pairs, `control_is_usable: True`** |

### A7 — what "clean" means here

`dropped_feature_type` reads **0**, and that is the correct post-fix state, not a
failure. The rewritten fetcher never *collects* UniProt's `Site` catch-all, so
there is nothing left downstream to drop. The number that shows the fix landed
is **`dropped_description`: 125 → 0** — the description heuristic that the code
itself called "a stopgap with known false negatives" now has nothing to guess
at. Every feature carries a `type`; the file holds 1,369 `Binding site` and 427
`Active site` annotations and nothing else.

### A7 — three silent wrong-protein bugs, found and fixed

None of these crashed. Each produced a real, reviewed UniProt entry for the
wrong protein, and the target then contributed a wrong or empty site list to
every average:

| Target | Was | Should be | How it was caught |
|---|---|---|---|
| `IKK-epsilon` | Q96MC9, "Putative uncharacterized protein IKBKE-AS1" (antisense transcript) | Q14164, the kinase | gene search took the first of `size=1` |
| `PRKCH` | C0HM02, "PRKCH upstream open reading frame 2", 52 aa | P24723, the kinase, 683 aa | uORF is filed under the *same* gene symbol — only sequence length separates them |
| `MST1` | P26927, macrophage-stimulating 1 (hepatocyte growth factor-like) | Q13043 = **STK4** | gene-symbol collision; normal length, exact symbol match, so only the "not described as a kinase in a kinase panel" sweep found it |

The fetcher now requests 10 candidates and picks on exact gene-symbol match then
longest sequence. All three checks live in
`python -m src.data.resolve_unmapped --dataset davis --audit`, which currently
flags one target (`CASK`, correct — UniProt just names it oddly).

Also fixed: `organism_id` → `taxonomy_id`, without which every non-human target
(`PFCDPK1`, `PKNB`, `PFPK5`) was unresolvable, because their reviewed entries
sit under *strain* taxa rather than the species id.

### A13 — why the control arm was rebuilt

The v2 plan's control arm is five antiviral proteins. `confound_report` gates
the control at **≥20 distinct non-kinase targets**, so five could never clear
it — that was true before any extraction problem. Two further findings:

- **SARS-CoV-2 cannot be extracted from BindingDB 2026-07-31.** All 18,149
  SARS-CoV-2 rows are filed under "Replicase polyprotein 1ab" carrying the full
  **7,096-residue** polyprotein. Mpro is residues 3264–3569; exactly 3 rows say
  so. Nothing separates an Mpro measurement from an RdRp one by target name, and
  7,096 residues sits almost entirely outside the 1,000-residue window anyway.
  Mpro and RdRp are now `OPTIONAL_TARGETS` with the reason recorded in code.
- That leaves three antiviral proteins, which are now a **named case study**
  rather than the control arm.

`src/data/build_nonkinase_panel.py` builds the real arm: 60 distinct human
non-kinase targets with UniProt binding-site annotation, inside the model's
window. **The filter that matters is not "is this a kinase" but "does this bind
a nucleotide"** — HSP90, DNA gyrase B, helicases, myosins and NADPH-dependent
oxidoreductases are non-kinases with nucleotide pockets, and admitting them
would put the confound inside the arm built to exclude it. Checked against
UniProt's annotated ligand, not the protein name.

**Open question for 124AD0067:** three panel targets have most of their
annotated sites on cotransport ions — `SLC6A3` (14/20), `SLC6A4` (10/16),
`DRD4` (2/4). No drug binds a sodium-coordination residue, so those positions
inflate precision@k the same way the `Site` catch-all did. Deliberately **not**
filtered: zinc cuts the other way, since carbonic anhydrase and HDAC inhibitors
chelate the catalytic zinc directly, so zinc *is* the drug site there. The
ligand is on every feature in `data/nonkinase_ground_truth_sites.json` so it can
be excluded or reported separately — that call is Track C's.

## Track B — 124AD0015 (Core model)

| | Item | Status |
|---|---|---|
| B1 | Drug encoder | ✅ tested, padding-invariant |
| B2 | Protein encoder, attention exposed | ✅ tested, zero weight on padding |
| B3 | Fusion + prediction head | ✅ end-to-end on dummy data |
| B4 | Training loop with checkpointing | ✅ verified `--dummy` |
| B5 | Trained on real DAVIS/KIBA, all four splits | ⏳ **no longer blocked — A4 is done.** The 24-run grid is now the critical path for the whole project |
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

None of these are code problems. They need data or compute — `rest.uniprot.org`
and `bindingdb.org` are unreachable from the environment the code was written
in, so every network step has to run on a team member's own machine.

| # | Item | Owner | Command | Est. |
|---|---|---|---|---|
| 0 | ~~Build the splits~~ | A | `python -m src.data.build_splits` | ✅ **done 2026-08-09** |
| 1 | ~~Antiviral subset~~ | A | `python -m src.data.extract_antiviral --from-cache` | ✅ **done** (3 targets; SARS-CoV-2 documented unavailable) |
| 2 | ~~Re-fetch ground truth with feature types~~ | A | `python -m src.data.fetch_binding_sites --dataset davis` / `kiba` | ✅ **done** |
| 3 | ~~KIBA gene map~~ | A | `python -m src.data.build_gene_map --dataset kiba` | ✅ **done**, 229/229 |
| 4 | ~~Unmapped DAVIS targets~~ | A | `python -m src.data.resolve_unmapped --dataset davis --apply` | ✅ **done**, 19 documented unresolvable |
| 5 | ~~Non-kinase control panel~~ | A | `python -m src.data.build_nonkinase_panel --scan/--select/--build` | ✅ **done**, 60 targets |
| 6 | **Train ColdSite-DTI: 2 datasets × 4 splits × 3 seeds** | **B** | `python -m src.model.run_grid --preflight`, then `python -m src.model.run_grid` | **24 runs, HPC — the critical path** |
| 7 | Fill the remaining 23 baseline cells | A | per `src/data/baselines/README.md` | the long pole |
| 8 | Decide how to treat cotransport-ion sites in the panel | C | see A13 above | a judgement call, not code |

**Everything Track A was blocking is now unblocked.** Item 6 is the critical
path for the entire project: no real precision@k number exists until checkpoints
do. Item 7 is Track A's remaining work and is the long pole overall.

Item 3 used to be a two-hour UniProt ID-mapping job. It is now one command,
because `fetch_binding_sites.py` records the gene symbol and protein name for
every target into `*_provenance.json` — the names were always inside the entries
it downloads, so the map falls out of item 2 with no second pass over the API,
and the symbols are guaranteed to come from the same UniProt snapshot as the
sites they stratify.

Verified download link for item 1 (BindingDB release 2026-07-31, 565 MB zipped,
~3–4 GB unzipped) — `curl -L` does not work in PowerShell, use this:

```powershell
$url = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
Invoke-WebRequest -Uri $url -OutFile "data\raw\BindingDB_All_tsv.zip"
Expand-Archive -Path "data\raw\BindingDB_All_tsv.zip" -DestinationPath "data\raw\" -Force
```

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
