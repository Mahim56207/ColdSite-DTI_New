# Runbook — 124AD0008, Track A

Every command you need, in order, with what "correct" looks like after each one.
Written for the **Anaconda Prompt**, which is `cmd`, not PowerShell. That
distinction matters in exactly two places and both have bitten this project
before:

- `$var = "..."` and `Invoke-WebRequest` are PowerShell-only. In `cmd` they fail
  with *"is not recognized as an internal or external command"*.
- `curl` in `cmd` is the **real** curl.exe that ships with Windows 10+, so
  `curl -L` works fine here. It is only in *PowerShell* that `curl` is aliased
  to `Invoke-WebRequest` and `-L` blows up with a `ParameterBindingException` —
  which is what happened last time and is logged as issue #22 in the handover.

Copy one block at a time; check the expected output before moving on.

Steps marked **[done]** were already run — the files are on your disk. Steps
marked **[you]** need your machine: they need the network (UniProt and BindingDB
are not reachable from the sandbox) or your GitHub credentials.

---

## Before anything: open the right shell

Open **Anaconda Prompt** (not plain PowerShell, not CMD), then:

```powershell
conda activate coldsite
cd "C:\Users\Uday Raj\OneDrive\Desktop\CODES\ColdSite-DTI"
python -c "import sys; print(sys.executable)"
```

The last line must print a path containing `envs\coldsite`. If it doesn't, you
are in the wrong Python and everything after this will install into the wrong
place — that is bug #20 in the handover, and it cost four rounds last time.

If `conda activate coldsite` errors with "environment does not exist":

```powershell
conda create -n coldsite python=3.11 -y
conda activate coldsite
python -m pip install -r requirements.txt
```

Always `python -m pip`, never bare `pip`.

---

## Step 0 — Connect the folder to GitHub  **[you]**

Your folder came from "Download ZIP", so it had no git history at all. `git init`
and the remote are now set up, but the folder has never spoken to GitHub. This
step attaches it to the real history so your commits land **on top of** what
124AD0015 has pushed, instead of replacing it.

```powershell
git fetch origin
```

You will be asked to sign in. Use a **Personal Access Token**, not your password:

1. Go to github.com → click your avatar (top right) → **Settings**
2. Scroll to the bottom of the left sidebar → **Developer settings**
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
4. Note: `ColdSite-DTI`. Expiration: 90 days. Tick the **`repo`** checkbox.
5. **Generate token** at the bottom, then copy the token — it is shown once.
6. When git asks for **Username**, type `udayraj1238`. When it asks for
   **Password**, paste the token.

Now align your working folder with the remote without touching any file:

```powershell
git reset --mixed origin/main
git status --short
```

`--mixed` changes only git's bookkeeping, never your files. What you should see
is a list of modified/new files. **Read that list.** If it shows files you did
not expect — especially anything under `src/model/` — your ZIP is older than
what 124AD0015 pushed, and committing everything would silently revert his work.
In that case stop and run:

```powershell
git diff --stat origin/main -- src/model/
```

If that prints anything, tell the team before committing. Nothing later in this
runbook stages `src/model/`, so as long as you follow the `git add` commands
exactly as written you cannot clobber him by accident.

Set your identity once, if you never have:

```powershell
git config user.name "Uday Raj"
git config user.email "rajuday6002@gmail.com"
```

---

## Step 1 — DeepDTA data files  **[done]**

`src/data/baselines/deepdta/data/{davis,kiba}/` now holds `ligands_can.txt`,
`proteins.txt` and `Y`. Verify:

```powershell
python -m src.data.load_data
```

Expected, exactly:

```
davis: 30056 measured pairs, 68 unique drugs, 442 unique targets, Y range [5.000, 10.796]
kiba: 118254 measured pairs, 2111 unique drugs, 229 unique targets, Y range [0.000, 17.200]
```

Those numbers match the published DeepDTA benchmark. If yours differ, you are
loading from somewhere else — do not continue.

These files are gitignored (`src/data/baselines/*`) and will not be committed.

---

## Step 2 — The four splits  **[done]** — this is what unblocked 124AD0015

```powershell
python -m src.data.build_splits
```

Already run. `data/splits/` now holds 8 split directories
(`{davis,kiba}` × `{random,cold_drug,cold_target,cold_pair}`), each with
`train.csv`, `valid.csv`, `test.csv`. Every leakage check reported clean.

| dataset | split | train | valid | test |
|---|---|---|---|---|
| davis | random | 21039 | 3006 | 6011 |
| davis | cold_drug | 21658 | 2652 | 5746 |
| davis | cold_target | 21080 | 2992 | 5984 |
| davis | cold_pair | 15190 | 264 | 1144 |
| kiba | random | 82778 | 11825 | 23651 |
| kiba | cold_drug | 83807 | 12073 | 22374 |
| kiba | cold_target | 85452 | 10701 | 22101 |
| kiba | cold_pair | 58041 | 1334 | 4375 |

The full table is in `results/split_summary.md`. That file is gitignored along
with everything else under `results/` — deliberately, so a placeholder number
can never sit in the repo looking like a finding — so paste the table into your
message to the team rather than expecting them to see the file.

**Two things to actually tell the team:**

- **Message 124AD0015 now.** His 24-run grid was blocked on exactly this and is
  no longer blocked. He needs `data/splits/` on his own machine — it is
  gitignored (417 MB), so he regenerates it by running the same command after
  putting the DeepDTA files in place. Point him at Step 1 and Step 2 here.
- **DAVIS cold-pair valid is 264 rows.** That is small enough that early-stopping
  on it will be noisy. It is inherent to requiring both drug and target unseen,
  not a bug — but he should know before he interprets a validation curve.
- **`data/splits/` is 417 MB inside your OneDrive folder.** It is gitignored, so
  it will never reach GitHub, but OneDrive will try to sync it. If that is a
  problem, right-click the `data\splits` folder → **Always keep on this device**
  is what you *don't* want; consider excluding it from sync.

---

## Step 3 — Re-fetch the ground truth, with feature types  **[you]** — ~30 min

This is the metric-validity fix. The committed files were built by a fetcher
that collected UniProt's `Site` catch-all, which carries protease cleavage
points and chromosomal breakpoints — positions no drug binds to, currently being
scored as correct answers. Roughly 136 DAVIS and 68 KIBA annotations.

Right now `ground_truth.py` filters them with a description-text heuristic,
which it documents as a stopgap with known false negatives. The re-fetch records
each feature's `type`, so the filter becomes exact.

```powershell
python -m src.data.fetch_binding_sites --dataset davis
```

Takes ~15 minutes (442 targets, caching collapses variants, 0.2s between calls).
It prints one line per target. Then:

```powershell
python -m src.data.fetch_binding_sites --dataset kiba
```

Another ~10 minutes for 229 targets. Now verify:

```powershell
python -m src.data.ground_truth
```

**The number that matters is `dropped_feature_type`.** It currently reads `0`
for both datasets, which is the signature of a file that predates type
recording. After a successful re-fetch it must be **greater than 0** — that is
the contamination being removed properly instead of guessed at. If it still
reads 0, the fetch did not overwrite the file; check that
`data/davis_ground_truth_sites.json` has a recent timestamp.

You should also now have two new files that did not exist before:

```powershell
dir data\*_provenance.json
```

`davis_ground_truth_sites_provenance.json` and the KIBA equivalent. Step 4 and
Step 5 both read them.

**If UniProt times out partway through:** just re-run the same command. It
starts over, which is slower but safe. Do not hand-edit the JSON.

### 3a. Spot-check the aliases — do not skip this

26 DAVIS targets resolve through `src/data/target_aliases.py`, which is a
human's assertion that two names mean the same protein. A wrong assertion does
not crash; it attaches another protein's binding sites to this target and scores
them as correct answers. List what came in that way:

```
python -c "import json; p=json.load(open('data/davis_ground_truth_sites_provenance.json')); print('\n'.join(f'{k:24s} {v[\"resolution\"]:28s} {v[\"uniprot_accession\"]:10s} {v[\"gene_name\"]} | {v[\"protein_name\"]}' for k,v in sorted(p.items()) if v['resolution'].startswith('alias')))"
```

Read the `protein_name` column. Every line should obviously be the kinase the
DAVIS name refers to. Two to look at hardest:

- **`OSR1`** — genuinely ambiguous. `OSR1` is also the symbol for the
  transcription factor *odd-skipped related 1*, which is not a kinase. In a
  kinase panel it should be **OXSR1**, oxidative-stress-responsive kinase 1. If
  the protein name comes back as a transcription factor, the alias is wrong.
- **`CDK4-cyclinD1` and `CDK4-cyclinD3`** — both deliberately resolve to CDK4
  and therefore get identical site lists. That is the same wild-type
  approximation already applied to the ABL1 point mutants. It is fine, but it
  has to be stated in Methods, not discovered by a reviewer.

Anything that looks wrong: delete that entry from `DAVIS_TARGET_ALIASES` and
resolve it by hand through Step 5's overrides file instead.

If any line reads `alias_not_found(SYMBOL)`, the symbol in the table is not one
UniProt recognises — fix the table, don't re-run and hope.

---

## Step 4 — Build the gene map, turn the KIBA control arm on  **[you]** — 1 min

This is the whole of your Priority 3, and it is now one command rather than two
hours of ID-mapping work.

The reason: `target_family.py` classifies kinase vs non-kinase by name. DAVIS
uses gene symbols (`ABL1`), so it classifies fine. KIBA uses UniProt accessions
(`O00141`), which carry no family signal, so all 224 KIBA targets read UNKNOWN
and the confound control cannot run on that dataset at all. The gene symbols
were always inside the UniProt entries the fetcher downloads — they just were
not being saved. Now they are, so the map falls out of Step 3 for free.

```powershell
python -m src.data.build_gene_map --dataset kiba
python -m src.data.build_gene_map --dataset davis
```

Expect `pct_with_gene_symbol` in the 90s. Then the payoff:

```powershell
python -m src.evaluation.target_family
```

**Before** (what it says today):

```
data/kiba_ground_truth_sites.json
  n_total  224   n_kinase  0   n_non_kinase  0   n_unknown  224
  control_is_usable  False
```

**After**, KIBA's `n_kinase` must no longer be 0. That is the check the Part 2
guide asks for.

Note what this does and does not fix. It makes KIBA *classifiable*, so kinases
are now identified on both datasets. It does **not** create non-kinase targets —
`control_is_usable` stays `False` until Step 6, because the only non-kinase
proteins in this project come from the antiviral subset. Steps 4 and 6 are two
halves of the same control.

---

## Step 5 — The unmapped DAVIS targets  **[you]** — ~2h, mostly manual

DAVIS has 442 targets; ground truth covers ~409. The other 33 are silently
absent from every precision@k average, and they are not missing at random — if
they are all fusion constructs or all non-human, that is a systematic bias in
the headline number that nothing downstream can detect.

See what actually failed and why:

```powershell
python -m src.data.resolve_unmapped --dataset davis --report
```

This groups them by failure route (`not_found`, `failed`, `no_sites`). Read the
grouping before you start looking things up — if 20 of them share one cause,
that is one fix, not twenty.

Generate the worksheet:

```powershell
python -m src.data.resolve_unmapped --dataset davis --template
```

That writes `data/davis_target_overrides.json`, one entry per failure. Open it
in a text editor. For each target:

1. Search the name at <https://www.uniprot.org> — tick **Reviewed (Swiss-Prot)**
   and set organism to **Homo sapiens** in the left-hand filters.
2. If you find it, copy the accession (looks like `P00533`) into
   `"uniprot_accession"`.
3. If it genuinely has no entry — some DAVIS names are domain-only constructs or
   fusions — put a short reason in `"unresolvable"` instead. **Do this rather
   than forcing a near-match.** A wrong accession attaches another protein's
   binding sites to this target, and that does not crash; it produces a
   plausible wrong precision@k, the same failure shape as the coordinate bug.
4. Re-running `--template` never overwrites work you have already filled in, so
   you can stop and come back.

Then apply:

```powershell
python -m src.data.resolve_unmapped --dataset davis --apply
python -m src.data.ground_truth
```

`targets_usable` should have gone up by however many you resolved. Everything
you marked unresolvable is counted and printed — **that number goes in the
Methods section**, so keep it.

---

## Step 6 — The antiviral subset  **[you]** — ~1h, mostly download

This is the control arm, and it is the highest-priority scientific gap in the
project. `data/processed/antiviral_clean.csv` is currently 614 rows and **100%
HIV-1 protease** — four of the five required targets are simply absent, so there
are zero usable non-kinase targets anywhere in the project.

### 6a. Download the file

The exact link, verified on the BindingDB downloads page (release 2026-07-31).
In the **Anaconda Prompt** (`cmd`), one line:

```
curl -L -o "data\raw\BindingDB_All_tsv.zip" "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
```

This works because `cmd` uses the real curl.exe. If you are in **PowerShell**
instead, that same command fails — there `curl` is an alias for
`Invoke-WebRequest`, which has no `-L` flag. The PowerShell form is:

```powershell
$url = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
Invoke-WebRequest -Uri $url -OutFile "data\raw\BindingDB_All_tsv.zip"
```

The zip is **565 MB**; it expands to roughly 3–4 GB. Check you have ~5 GB free
on C: first. It takes 5–20 minutes depending on your connection.

If the download dies partway, delete the partial file and re-run — do not try to
resume it.

### 6b. Unzip

In `cmd`, `Expand-Archive` does not exist either. Use tar, which ships with
Windows 10+ and reads zips:

```
tar -xf "data\raw\BindingDB_All_tsv.zip" -C "data\raw"
dir data\raw\*.tsv
```

Note there is **no trailing backslash** inside the closing quote on `-C`. In
`cmd`, `"data\raw\"` ends in `\"`, which escapes the quote rather than closing
it, and tar reports `could not chdir to 'data\raw"'`.

Note the exact filename it produces — it will be something like
`BindingDB_All.tsv`. Use that name in the next command.

### 6c. Extract the five targets

```powershell
python -m src.data.extract_antiviral --source data\raw\BindingDB_All.tsv
```

**If it refuses to write because a target is missing — good, that is the script
working.** Do *not* add `--allow-partial`. That flag is for a deliberate partial
run you are going to document, not for making a red message go away, and a
half-built control arm is worse than a stated limitation because it looks
complete.

Instead, find the spelling your BindingDB release actually uses:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', usecols=['Target Name'], nrows=2000000, on_bad_lines='skip'); n=d['Target Name'].dropna().unique(); import re; print([x for x in n if re.search(r'neuraminidase|reverse transcriptase|main protease|3C-like|nsp5|nsp12|RNA-directed', str(x), re.I)][:60])"
```

Add the real spelling to `TARGET_PATTERNS` in
`src/data/extract_antiviral.py`, **and** add a matching case to
`tests/test_antiviral.py` so the pattern is pinned. Then re-run.

### 6d. What the antiviral file is now for

The five antiviral targets are a **named case study**, not the control arm. Two
findings forced that change, and both belong in the Methods section:

- **`confound_report` gates the control at ≥20 distinct non-kinase targets.**
  The master plan's arm is five proteins. It could never clear its own
  threshold — that was true before any extraction problem.
- **SARS-CoV-2 cannot be extracted from this BindingDB release.** All 18,149
  SARS-CoV-2 rows are filed under "Replicase polyprotein 1ab" carrying the full
  **7,096-residue** polyprotein; Mpro is residues 3264–3569 and exactly 3 rows
  say so. Nothing separates an Mpro measurement from an RdRp one by target
  name, and 7,096 residues sits almost entirely outside the model's 1,000-
  residue window anyway. Mpro and RdRp are now `OPTIONAL_TARGETS` with the
  reason recorded in the code — a documented scope reduction, not a suppressed
  error. The specs stay in place so a future release that names the domains
  properly gets picked up automatically.

So the file should now build with HIV-1 protease, HIV-1 RT and influenza
neuraminidase. The real control arm is Step 9.

---

## Step 9 — The non-kinase control panel  **[you]** — ~40 min

This is what actually answers the kinase confound.

The question the control has to settle is not "do viral proteins behave
differently". It is: **does the cold-target result survive on proteins that do
not share the ATP pocket the model saw several hundred times in training?** Any
well-annotated non-kinase answers that, so the arm is widened past the five
antivirals until it clears the gate.

### 9a. Scan for per-accession pair counts (~20 min)

```
python -m src.data.build_nonkinase_panel --scan --source data\raw\BindingDB_All.tsv
```

One pass over the 9 GB file, keeping accession + SMILES + affinity only. No
sequences — those come from UniProt in the next step, deliberately: the
binding-site coordinates are UniProt's, and a coordinate is meaningless against
a different sequence. BindingDB's chains are frequently tagged or truncated
constructs, so pairing its sequence with UniProt numbering would offset every
index silently. Same failure as the 1-indexed/0-indexed bug in the handover.

### 9b. Select the panel (~10 min, needs UniProt)

```
python -m src.data.build_nonkinase_panel --select
```

Fetches the busiest accessions and keeps only those that are non-kinase, have
real binding-site annotation, and fit inside the 1,000-residue window.

**The filter that matters most rejects non-kinases.** HSP90, DNA gyrase B,
helicases, myosins and NADPH-dependent oxidoreductases are not kinases but do
bind nucleotides. Letting them in would put the very thing being controlled for
inside the control arm, and the stratified panel would then show "no difference
between families" for a reason with nothing to do with the paper's claim. The
check runs against UniProt's annotated **ligand**, not the protein's name,
because the name does not reliably say — "Heat shock protein HSP 90-alpha"
mentions no nucleotide anywhere.

Read the rejection counts it prints. `binds_nucleotide` should be a large
bucket; if it is near zero, the ligand field is not being read and the filter is
doing nothing.

### 9c. Build it

```
python -m src.data.build_nonkinase_panel --build
```

Writes `data/processed/nonkinase_panel.csv`, its ground truth, provenance, and
the family assignments. It prints the confound report and states plainly whether
the ≥20 gate passes.

**If it comes up short**, don't lower the gate. Raise `--limit` (more candidate
accessions) or `--panel-size`, and re-run `--select`. The gate is 20 because
below that the stratified comparison has no power, and a number a reviewer
cannot believe is worse than a stated limitation.

### 9d. Sanity-check the panel by eye

```
python -c "import json; d=json.load(open('data/nonkinase_panel_targets.json')); print(len(d),'targets'); [print(f\"  {v['gene_name']:10s} {v['n_features']:>2} sites {v['sequence_length']:>4} aa  {v['protein_name'][:44]}\") for v in list(d.values())[:30]]"
```

Every line should be a protein you would defend as a non-kinase, non-nucleotide
drug target. Anything you would not, delete from the JSON and re-run `--build`.

### 6e. Delete the guard test

```powershell
python -m pytest tests\test_antiviral.py -q
```

`test_committed_antiviral_file_is_still_incomplete` is *supposed* to fail once
the file is fixed — it is a deliberate tripwire on the broken artefact. Delete
that one test from `tests/test_antiviral.py`, then re-run the suite.

---

## Step 7 — Housekeeping

Delete the superseded ground-truth file. The new
`data/davis_ground_truth_sites.json` (409+ targets, keyed by DAVIS `Target_ID`)
is strictly better than the old `data/processed/binding_sites_ground_truth.json`
(350 targets, keyed by UniProt accession). Leaving both means someone
eventually points the evaluation at the stale, more-contaminated one and gets
numbers nobody can explain.

```
del "data\processed\binding_sites_ground_truth.json"
```

Do this **only after Step 3 succeeded**.

---

## Step 8 — Verify everything, then push  **[you]**

Full suite:

```powershell
python -m pytest tests\ -q
```

Expected: all pass. It takes ~10 seconds. If
`test_an_untrained_model_scores_around_chance_not_above_it` fails, **stop** —
that is the sanity floor, and it firing means the metric pipeline is measuring
an artefact and every number produced after it is suspect.

Then the coverage report one last time:

```powershell
python -m src.data.ground_truth
python -m src.evaluation.target_family
```

Now commit. Stage explicitly — never `git add .` here, because that would sweep
in whatever else differs between your ZIP and the remote:

```powershell
git add src/data/fetch_binding_sites.py src/data/build_gene_map.py src/data/resolve_unmapped.py
git add src/evaluation/target_family.py tests/test_gene_map.py
git add docs/RUNBOOK_124AD0008.md
git add data/davis_ground_truth_sites.json data/kiba_ground_truth_sites.json
git add data/*_provenance.json data/*_uniprot_to_gene.json data/davis_target_overrides.json
git add data/processed/antiviral_clean.csv
git add STATUS.md
git status --short
```

Read the `git status --short` output before committing. Everything listed should
be something you meant to change.

```powershell
git commit -m "Track A: splits built, ground truth re-fetched with feature types, KIBA gene map, antiviral control arm"
git push origin main
```

If push is rejected with "fetch first", someone pushed while you were working:

```powershell
git pull --rebase origin main
git push origin main
```

---

## What to tell the team afterwards

- **124AD0015:** splits exist, his 24-run grid is unblocked. He regenerates
  `data/splits/` locally via Steps 1–2. Flag the 264-row DAVIS cold-pair
  validation set to him.
- **124AD0067:** the confound control now runs on both datasets. Whether it is
  *sufficient* is her call — that is her Priority 2, and the answer depends on
  the distinct non-kinase count from Step 6d.

---

## Quick reference

| # | Command | Needs network | Time |
|---|---|---|---|
| 1 | `python -m src.data.load_data` | no | instant |
| 2 | `python -m src.data.build_splits` | no | ~2 min |
| 3 | `python -m src.data.fetch_binding_sites --dataset davis` (and `kiba`) | UniProt | ~30 min |
| 4 | `python -m src.data.build_gene_map --dataset kiba` | no | instant |
| 5 | `python -m src.data.resolve_unmapped --dataset davis --report/--template/--apply` | UniProt (apply only) | ~2h manual |
| 6 | `python -m src.data.extract_antiviral --source data\raw\BindingDB_All.tsv` | BindingDB (download) | ~1h |
| — | `python -m src.data.ground_truth` | no | instant |
| — | `python -m src.evaluation.target_family` | no | instant |
| — | `python -m pytest tests\ -q` | no | ~10s |
