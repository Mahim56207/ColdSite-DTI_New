#!/usr/bin/env bash
# ===================================================================
#  The week's grid: DAVIS, binary, 3 models x 4 splits x 3 seeds.
#
#  Why this file exists alongside the .bat runners
#  -----------------------------------------------
#  The .bat files only run on Windows. The grid runs on Kaggle, Colab
#  or the HPC, all of which are Linux, and none of which can launch a
#  batch file.
#
#  Why DAVIS only, and why binary
#  ------------------------------
#  KIBA is 118,254 pairs against DAVIS's 30,056 -- about 4x the
#  training time for the same 36 cells, which does not fit the window.
#  Binary is the only task all the audited models share: HyperAttentionDTI
#  is Linear(512, 2) + CrossEntropyLoss and cannot do regression at all.
#  See results.md.
#
#  Batch sizes are set for a 16 GB card (T4, P100). Measured peak
#  activation memory, forward + backward, 1000-residue protein:
#
#    DeepDTA            batch 256 -> 0.7 GB
#    ColdSite-DTI       batch  64 -> 8.7 GB   (16 -> 2.3 GB, 32 -> 4.4 GB)
#    HyperAttentionDTI  batch  32 -> ~5 GB    (8 -> 1.7 GB)
#
#  On a 4 GB card: COLDSITE_BATCH=16 HAT_BATCH=8 HAT_ACCUM=4 ./run_davis_grid.sh
#  HAT_BATCH x HAT_ACCUM is the effective batch and must stay at the
#  vendored 32 -- accumulation changes the memory, not the gradient.
#
#  Every runner is resumable: finished cells are skipped, so a killed
#  Kaggle session costs only the cell it was on.
#
#  Usage:
#     ./run_davis_grid.sh              # all three models
#     ./run_davis_grid.sh deepdta      # one model
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-1 2 3}"
SPLITS="${SPLITS:-random cold_drug cold_target cold_pair}"
EPOCHS="${EPOCHS:-100}"
COLDSITE_BATCH="${COLDSITE_BATCH:-64}"
HAT_BATCH="${HAT_BATCH:-32}"
HAT_ACCUM="${HAT_ACCUM:-1}"
DEEPDTA_BATCH="${DEEPDTA_BATCH:-256}"
# Checkpoint-selection floor. Applied identically to all three models on
# purpose: the audit compares them to each other, and a model checkpointed at
# epoch 2 against one at epoch 16 confounds the explanation axis with training
# volume. See src/model/early_stopping.py.
MIN_EPOCHS="${MIN_EPOCHS:-10}"

WHICH="${1:-all}"

run_deepdta() {
  echo "=== DeepDTA (binary), DAVIS ==="
  for seed in $SEEDS; do
    for split in $SPLITS; do
      python -m src.model.train_deepdta \
        --split-dir "data/splits/davis/$split" \
        --dataset davis --split "$split" \
        --task binary --seed "$seed" \
        --batch-size "$DEEPDTA_BATCH" --min-epochs "$MIN_EPOCHS" \
        --epochs "$EPOCHS" --skip-if-done
    done
  done
}

run_coldsite() {
  echo "=== ColdSite-DTI (binary), DAVIS ==="
  # run_grid validates the first cell end to end before launching the rest,
  # and refuses to start against missing or colliding cells.
  python -m src.model.run_grid \
    --datasets davis \
    --splits "$(echo $SPLITS | tr ' ' ',')" \
    --seeds "$(echo $SEEDS | tr ' ' ',')" \
    --task binary --epochs "$EPOCHS" \
    --batch-size "$COLDSITE_BATCH" --min-epochs "$MIN_EPOCHS"
}

run_hyperattentiondti() {
  echo "=== HyperAttentionDTI (binary), DAVIS ==="
  echo "    the long job: expect ~7-19 h on a T4 for 12 cells"
  for seed in $SEEDS; do
    for split in $SPLITS; do
      python -m src.model.train_hyperattentiondti \
        --split-dir "data/splits/davis/$split" \
        --dataset davis --split "$split" \
        --seed "$seed" \
        --batch-size "$HAT_BATCH" --accum-steps "$HAT_ACCUM" \
        --min-epochs "$MIN_EPOCHS" \
        --epochs "$EPOCHS" --skip-if-done
    done
  done
}

case "$WHICH" in
  deepdta)            run_deepdta ;;
  coldsite)           run_coldsite ;;
  hyperattentiondti)  run_hyperattentiondti ;;
  all)
    # cheapest first: a shape error surfaces in minutes rather than hours
    run_deepdta
    run_coldsite
    run_hyperattentiondti
    ;;
  *)
    echo "unknown model '$WHICH'." >&2
    echo "usage: $0 [all|deepdta|coldsite|hyperattentiondti]" >&2
    exit 2
    ;;
esac

echo
echo "Grid complete. Next:"
echo "  python -m src.evaluation.run_faithfulness --dataset davis --seed 1 --task binary"
echo "  python -m src.evaluation.run_ladder --dataset davis --seed 1 --task binary \\"
echo "      --ground-truth data/davis_ground_truth_sites.json \\"
echo "      --accuracy-json results/accuracy_davis_seed1.json"
