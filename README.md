# ColdSite-DTI

**Does a drug–target interaction model's explanation stay trustworthy when the drug or protein is one it has never seen before?**

We build a CNN-BiLSTM + multi-head-attention drug–target interaction (DTI) predictor, in the style of Dr. Chandra Mohan Dasari's own published transcription-factor binding-site architecture, and measure — for the first time as a single, connected result — how the *explanation quality* (does attention point at real binding sites?) degrades across four levels of "how unfamiliar is this drug/protein to the model," from a standard random split to a fully unseen drug-and-protein pair. We validate on a real-world antiviral case study (SARS-CoV-2 Mpro/RdRp, HIV protease/RT, influenza neuraminidase).

Full context, motivation, and the complete project plan: **[`docs/00_MASTER_PLAN.md`](docs/00_MASTER_PLAN.md)** — read this first.

## Team

| Roll number | Track | Owns |
|---|---|---|
| 124AD0008 | A — Data, cold-start splits, baselines | [`docs/01_GUIDE_124AD0008.md`](docs/01_GUIDE_124AD0008.md) |
| 124AD0015 | B — Core model architecture & training | [`docs/02_GUIDE_124AD0015.md`](docs/02_GUIDE_124AD0015.md) |
| 124AD0067 | C — Explanation testing, case study, literature | [`docs/03_GUIDE_124AD0067.md`](docs/03_GUIDE_124AD0067.md) |

**Supervisor:** Dr. Chandra Mohan Dasari · **Course:** Bioinformatics, B.Tech AI & Data Science, 5th Semester

## Repo structure

```
ColdSite-DTI/
├── docs/               Master plan + each person's individual guide
├── data/
│   ├── raw/             Downloaded datasets (gitignored — regenerate, don't commit)
│   └── splits/          The four difficulty splits (gitignored — regenerate, don't commit)
├── src/
│   ├── data/             Track A: loading, splitting, binding-site ground truth, baselines
│   ├── model/            Track B: drug/protein encoders, fusion, training loop
│   └── evaluation/       Track C: precision@k, significance testing, plots
├── notebooks/            Exploratory / scratch notebooks
├── results/              Output tables and figures (gitignored — regenerate, don't commit)
├── paper/                Paper drafts
├── requirements.txt
└── LICENSE
```

## Quickstart

```bash
git clone <this-repo-url>
cd ColdSite-DTI
python -m venv venv && source venv/bin/activate    # or use the college HPC's own environment setup
pip install -r requirements.txt
```

Then open your own guide in `docs/` and start with Part 1 — every track can begin immediately without waiting on the others.

Run the test suite:

```bash
python -m pytest tests/ -q          # 540 tests
python -m src.data.ground_truth     # ground-truth coverage report
```

## Reading the binding-site ground truth

Never parse the ground-truth JSON directly. UniProt annotations are 1-indexed
inclusive ranges; `precision_at_k` expects 0-indexed positions, and mixing them
returns a wrong number rather than an error. Always go through the adapter:

```python
from src.data.ground_truth import load_site_sets
sites = load_site_sets("data/davis_ground_truth_sites.json", max_len=1000)
```

See [`data/GROUND_TRUTH_README.md`](data/GROUND_TRUTH_README.md).

## Status

🟡 **Part 1 in progress** — August 2026. Solo code for all three tracks is
complete and tested; five items remain, all blocked on data downloads or
compute rather than code. Current state, what changed, and the exact remaining
commands: **[`STATUS.md`](STATUS.md)**.

See `docs/00_MASTER_PLAN.md` §6 for the full timeline (target: full draft by
mid-November 2026).
