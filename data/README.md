# Data

Nothing in this folder (other than this README) is tracked by git — datasets are regenerated locally, not committed, because they're large and are downloadable in seconds from their original sources. This keeps the repo small and fast to clone.

## How to regenerate everything

```bash
python -m src.data.load_data      # sanity-checks DAVIS/KIBA load correctly
python -m src.data.build_splits   # builds all four splits for both datasets into data/splits/
python -m src.data.binding_sites  # fetches binding-site ground truth (edit the ID list first)
```

## What goes where

- `raw/` — anything downloaded directly (e.g. the antiviral-filtered BindingDB subset CSV built in `docs/01_GUIDE_124AD0008.md` Step 3).
- `splits/` — the output of `build_splits.py`: twelve CSV files per dataset (4 split types × train/valid/test), plus `binding_sites.json` from `binding_sites.py`.

If you're picking this project up fresh (e.g. on the college HPC), just run the three commands above in order and this folder rebuilds itself.
