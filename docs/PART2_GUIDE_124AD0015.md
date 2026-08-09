# Part 2 Guide — 124AD0015
## Track B: Model, Training Grid, and Faithfulness

Read `docs/00_MASTER_PLAN_V2.md` first.

---

## What changed for you

Your Part 1 is done and it is the strongest work in the repo — all four
smoke tests pass, attention is exposed and masked correctly, and the padding
guarantees Track C depends on are pinned by 23 contract tests.

Two things change.

**Your model is no longer the contribution.** Under the audit framing
ColdSite-DTI is one subject among several. This is not a demotion — it means
nobody will reject the paper because ColdSite-DTI didn't beat MolTrans by 0.01
CI. Stop tuning for accuracy. The model needs to be *a solid, comparable
predictor*, nothing more.

**Faithfulness is now yours and it is core.** v1 listed masking as an optional
October stretch task. It is now a headline measurement, and it is a direct
extension of the model you already built, so it belongs on your track.

---

## Priority 1 — The training grid (BLOCKING for everyone, ~1 week HPC)

Not one model per split. **Three seeds per cell.**

```bash
for seed in 1 2 3; do
  for dataset in davis kiba; do
    for split in random cold_drug cold_target cold_pair; do
      python -m src.model.train \
        --split-dir data/splits/$dataset/$split \
        --dataset $dataset --split $split \
        --task regression --seed $seed \
        --epochs 100
    done
  done
done
```

That is 24 runs. Start with one cell end-to-end before launching the loop —
a shape error discovered on run 23 costs a week.

Checkpoint naming already includes dataset and split. **Add the seed**, or you
will overwrite runs and not notice. The most expensive mistake available in
this project is still reporting a cold-target number produced by a model
trained on cold-drug.

Record accuracy per run (`results/{tag}_results.json` already does this) and
hand the paths to Track C.

## Priority 2 — Wire up faithfulness (~2 days)

`src/evaluation/faithfulness.py` is built and tested. Your job is running it on
real checkpoints and interpreting the output.

```python
from src.evaluation.faithfulness import batch_faithfulness

summary = batch_faithfulness(model, drugs, proteins, attentions,
                             k=10, n_random_trials=5, max_pairs=200)
print(summary["comprehensiveness_delta"])
print(summary["explanation_is_load_bearing"])
```

**The only number that matters is the delta**, not the raw comprehensiveness.
Masking anything shifts the input off the training distribution and the
prediction moves for reasons unrelated to the explanation — this is the ROAR
critique, and it is why every metric in that module ships with a
random-masking control. A comprehensiveness of 0.4 means nothing on its own; it
means something next to a random-masking comprehensiveness of 0.05.

Budget: each pair costs `2 + 2×n_random_trials + len(k_values)` forward passes.
Use `max_pairs` — the mean over a few hundred pairs is the number, not the mean
over all of them.

**Interpreting the result.** If `comprehensiveness_delta ≤ 0` on the cold
splits, the model's explanations are not load-bearing there — the attention is
decoration. That is not a failed experiment. That is the paper's most
interesting possible finding, and it is exactly the "plausible but not
faithful" cell from the master plan. Report it plainly.

## Priority 3 — Two decisions only you can make (~1 day, do this week)

Both are Methods decisions the code deliberately leaves open. Both change the
headline number. Pick, write down the reasoning, tell the team.

**Truncation.** The model reads at most 1000 residues. At that setting 283
DAVIS and 165 KIBA annotated positions fall outside the window. That affects
**24 DAVIS and 14 KIBA targets**, of which **15 and 8** lose every annotation
and are dropped entirely. `truncation="exclude"` drops them; `"keep"` retains
them and lowers the ceiling. See `data/GROUND_TRUTH_README.md`.

**Cold-pair training volume.** Cold-pair discards every row whose drug is held
out but whose target is not. Two different ratios follow, and they must not be
mixed up: it **trains** on roughly **71%** of the pairs the other levels get,
and **uses** roughly **54%** of all measured pairs once the discarded rows are
counted. The training-volume confound is the first number. If that is not
stated, the level-4 drop reads as pure difficulty when part of it is simply
less training data. Either report the counts prominently, or subsample the
other three splits to match. Your call, but it must be a call.

> **Figures corrected 2026-08-09 (124AD0015).** This section previously read
> "dropping 15 and 10 targets entirely" and "roughly 54% of the pairs".
> Regenerate the truncation figures with `python -m src.data.ground_truth`
> (after the patch adding `targets_affected_by_truncation` /
> `targets_dropped_by_truncation` lands) and the volume figures from
> `results/split_summary.md`. KIBA's earlier "10" came from a counter that also
> included 2 targets lost to description filtering rather than to the window;
> the earlier "54%" was the pairs-*used* fraction quoted as the training ratio,
> which overstates the confound by roughly a factor of two.

## Priority 4 — Register your model, help with the others (~2 days)

`ColdSiteDTIAdapter` already exists in `src/evaluation/model_registry.py`.
124AD0008 owns the baseline adapters, but HyperAttentionDTI and MolTrans both
expose attention in ways closer to your expertise than theirs. Expect to pair
on those.

The contract is two methods and one trap: `explain()` must return exactly one
weight per **real** residue. Returning padded length does not crash — it
misaligns every ground-truth index past the cut.

## Definition of done

- [ ] 24 training runs complete (2 datasets × 4 splits × 3 seeds), checkpoints
      named with seed
- [ ] Accuracy recorded per run and handed to Track C
- [ ] Faithfulness run on every split, deltas reported against random control
- [ ] Truncation policy decided, written down, communicated
- [ ] Cold-pair volume confound handled (reported or matched)
- [ ] `validate_adapter` passing for ColdSite-DTI

## Common mistakes

- **Chasing accuracy.** Your model's job is to be comparable. Weeks spent on
  +0.01 CI are weeks not spent on faithfulness, which is the actual
  contribution.
- **Reporting raw comprehensiveness.** Always the delta.
- **One seed "to see if it works".** Fine as a smoke test, never as a result.
- **Calling `explain()` in train mode.** It forces `.eval()` itself, but if you
  write your own extraction path, attention dropout will perturb the weights.

## Where to get help

If a training run diverges or the loss goes NaN, gradient clipping is already
in the loop at max_norm 5.0 — check the split file first, not the model.
