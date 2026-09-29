# Optimizer probe (D4): is seed variance an artifact of AdamW? — SPECIFIED, NOT RUN

**Status (2026-09-29).** No checkpoint in the repository was trained with any optimizer other than AdamW. Every one of the 84 trained
models uses the published recipes: `src/model/train_hyperattentiondti.py` hard-codes AdamW (lr 5e-5, weight decay 1e-4 on weights and 0
on biases) with a cyclic learning rate, and its docstring records why: retraining a subject under another optimizer measures a model its
authors never released. There is no `--optimizer` flag and no other-optimizer or other-schedule result to inspect, so D4 cannot be
answered from existing files. This page specifies the probe; running it needs a decision, because it changes training code and spends
about 3 GPU-hours.

## What it would test, and what it would not

Question: does the between-seed disagreement of the attention map (cross-seed rank correlation, top-10 overlap, precision@10 spread)
persist when the same model is trained with plain SGD with momentum instead of AdamW?
* If the disagreement persists, it is not specific to Adam's per-parameter scaling. That is the only thing the probe can show.
* If it shrinks, that is a finding about the optimizer, not a refutation of the audit: the audit is about the published recipes.
* One SGD model per seed is a probe, not a subject. It joins no Holm family, no table of the paper, and no claim of the draft.
* Two SGD seeds are the minimum for a seed comparison; a single SGD run cannot say anything about seed variance (there is nothing to compare it with).

## The patch (small, off by default; NOT applied)

In `src/model/train_hyperattentiondti.py`:
1. `parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")`, `--momentum` (default 0.9).
2. Add `"optimizer"` (and `"momentum"`) to `RESUME_KEYS` so a resumed run cannot switch optimizer mid-cell.
3. Where the optimizer is built: `adamw` keeps today's code byte for byte; `sgd` builds
   `torch.optim.SGD([{params: weight_p, weight_decay: hp.weight_decay}, {params: bias_p, weight_decay: 0}], lr=args.lr, momentum=args.momentum)`
   with the same two parameter groups and the same `CyclicLR` (`cycle_momentum=True` is then required by SGD's momentum, or keep `False`
   and say so).
4. A test: the default path builds the identical optimizer state to before (compare `state_dict` after one step on a fixed seed), and
   `sgd` builds `torch.optim.SGD`.
The learning rate cannot be carried over: AdamW's 5e-5 would barely move SGD. A short, declared sweep is needed first (for example
lr in {1e-3, 1e-2, 1e-1} for two epochs on DAVIS random, seed 1, judged on validation AUROC only, before any attention is looked at).

## Command once patched (DAVIS random, HyperAttentionDTI, seeds 1 and 2)

```bash
python -u -m src.model.train_hyperattentiondti --split-dir data/splits/davis/random --dataset davis --split random --seed 1 \
  --batch-size 32 --accum-steps 1 --patience 15 --min-epochs 10 --epochs 100 --optimizer sgd --lr <swept> \
  --checkpoint-dir results/sgd_probe --results-dir results/sgd_probe
```
Repeat with `--seed 2`. Cost: about 6 min per epoch on a T4 (CLAUDE.md §4), at least 25 epochs: roughly 2.5-3 GPU-hours per seed, so
about 5-6 GPU-hours for two seeds, plus the sweep (about 0.5 h). In `quota.py` terms that is up to 6.5 of an account's 25.5 usable
GPU-hours for the week: run it on an account that is not being used for Wave A, and declare it with `--add-external`.

## Analysis (once trained)

`python scripts/certification/step9_ironclad_diagnostics.py`, pointed at the two SGD checkpoints: cross-seed attention rank correlation,
top-10 overlap and precision@10 for the two SGD seeds, beside the AdamW seeds 1 and 2 on the same proteins (D1c), plus the feature
sparsity table (D1d): whether the dead-channel collapse also appears under SGD is the most informative single number.

## Decision needed

Approve (1) the patch to the published-recipe trainer, and (2) about 6 GPU-hours on a spare account. Until then D4 stays unanswered,
and the report says so.
