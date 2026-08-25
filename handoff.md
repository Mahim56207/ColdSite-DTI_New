Repo: https://github.com/Mahim56207/ColdSite-DTI_New (my fork of udayraj1238/ColdSite-DTI).
Local clone: /Users/mahimagarwal/Documents/ColdSite-DTI, branch project-completion (clean, in sync).
Remotes: origin = my fork, upstream = udayraj1238/ColdSite-DTI. Both main and
project-completion are in sync with the fork; nothing is unpushed.
gh is authenticated as Mahim56207 with repo scope, so you can commit and push
directly — but confirm with me before any push.

FRAMING — read docs/00_MASTER_PLAN_V2.md and STATUS.md before anything else.
This is an AUDIT of published DTI interpretability claims, not a single-model
paper. All code is written and covered by 393 passing tests. What remains is GPU
work plus some data gaps on the audit side.

STATE OF THE BLOCKER (item 5 in STATUS.md) — SOLVED LOCALLY.
The 6 DeepDTA source files (ligands_can.txt, proteins.txt, Y — for davis and
kiba) are downloaded from hkmztrk/DeepDTA into src/data/baselines/deepdta/data/,
splits are built, all six cold-split leakage checks pass, 393 tests pass, and
  python -m src.model.run_grid --preflight
reports ready_to_launch: True.

BUT both dirs are gitignored (data/splits is 417MB), so they live on my Mac only
and do NOT travel via git. On Colab you must regenerate them first:
  mkdir -p src/data/baselines/deepdta/data/{davis,kiba}
  for ds in davis kiba; do for f in ligands_can.txt proteins.txt Y; do
    curl -sL https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/$ds/$f \
      -o src/data/baselines/deepdta/data/$ds/$f; done; done
  python -m src.data.build_splits
  python -m src.model.run_grid --preflight   # expect ready_to_launch: True
Then persist splits + source files to Drive so later Colab sessions skip the
rebuild. Sanity figures to expect:
  davis  30056 measured pairs, 68 drugs, 442 targets, Y range [5.000, 10.796]
  kiba  118254 measured pairs, 2111 drugs, 229 targets, Y range [0.000, 17.200]
DAVIS Y is converted in-code to pKd via -log10(Y/1e9), so use the raw-Kd files.

TWO NUMBERS THAT ARE EASY TO GET WRONG — both re-verified against the freshly
built splits, do not let them drift:
  - The grid is 2 datasets x 4 splits x 3 TRAINING seeds = 24 runs, NOT 72. The
    seeds vary weight init and batch order on one fixed split per cell. So seed
    error bars measure initialisation variance, not split-selection variance —
    that belongs in the paper.
  - Cold-pair TRAINS on ~71% of the pairs the other levels get (davis 72%, kiba
    70%) but USES ~54% of all measured pairs once discarded rows are counted
    (davis 55%, kiba 54%). Quoting 54% as the training ratio overstates the
    confound by about 2x. pct_of_largest_split reports the second number.

FIRST REAL TASK: rebuild the data on Colab per above, then launch the 24
training runs on GPU. After that, faithfulness on real checkpoints and the
volume-matched sensitivity run unblock.

STILL OPEN (audit side, none of these block training):
  - The kinase/non-kinase CONTROL ARM DOES NOT EXIST — non-kinase count is 0 for
    both datasets. KIBA uses UniProt accessions with no gene mapping. STATUS.md
    calls this the highest-priority gap in the project.
  - Antiviral rebuild: needs 5 targets with >=20 non-kinase; only 1 of 5 is in
    the committed file. Blocks the whole control arm.
  - Ground-truth re-fetch with feature types (metric validity).
  - 33 unmapped DAVIS targets (coverage).
  - Baseline adapters for DeepDTA / HyperAttentionDTI / MolTrans are registered
    stubs that raise with instructions until filled in.
  - Results table is 1 of 24 cells.

Colab is set up and you have browser access to it. notebooks/01_data_prep.ipynb
is the existing notebook.
