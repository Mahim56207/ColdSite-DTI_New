# Data

Large artefacts are regenerated locally rather than committed. The ground-truth
JSONs are the exception — they are small, slow to rebuild (one UniProt request
per target), and every track depends on them, so they stay tracked.

## Regenerating everything

```bash
# 0. source data for DAVIS/KIBA (not committed -- DeepDTA's own release files)
git clone https://github.com/hkmztrk/DeepDTA src/data/baselines/deepdta

# 1. sanity-check the datasets load
python -m src.data.load_data

# 2. build all four splits for both datasets + the summary table
python -m src.data.build_splits

# 3. binding-site ground truth (needs rest.uniprot.org)
python -m src.data.fetch_binding_sites --dataset davis
python -m src.data.fetch_binding_sites --dataset kiba

# 4. verify the ground truth converts cleanly for Track C
python -m src.data.ground_truth

# 5. antiviral subset (needs BindingDB_All.tsv, ~3GB, from bindingdb.org)
python -m src.data.extract_antiviral --source data/raw/BindingDB_All.tsv
```

Step 5 refuses to write a file unless all five required targets are present.
That is deliberate — the previous version wrote a single-target file and
reported success.

## What goes where

| path | contents | tracked? |
|---|---|---|
| `raw/` | bulk downloads (BindingDB TSV, etc.) | no |
| `splits/{dataset}/{split}/{train,valid,test}.csv` | output of `build_splits.py` | no |
| `processed/antiviral_clean.csv` | output of `extract_antiviral.py` | yes |
| `*_ground_truth_sites.json` | output of `fetch_binding_sites.py` | yes |
| `*_ground_truth_sites_provenance.json` | which UniProt accession each target resolved to | yes |

## Reading the ground truth

See **[`GROUND_TRUTH_README.md`](GROUND_TRUTH_README.md)**. Short version: never
parse the JSON directly, always go through `src.data.ground_truth.load_site_sets`.
UniProt is 1-indexed and the metric is 0-indexed, and getting that wrong returns
a wrong number rather than an error.
