# Baselines (124AD0008)

This folder is where the three comparison models live, so ColdSite-DTI has fair numbers to be measured against.

Clone each one into its own subfolder here:

```bash
cd src/data/baselines
git clone <DeepDTA repo url> deepdta
git clone <HyperAttentionDTI repo url> hyperattentiondti
git clone <MolTrans repo url> moltrans
```

Search "DeepDTA github", "HyperAttentionDTI github", and "MolTrans github" — the official repo is normally the top result, under the original paper's authors. Double check each repo actually references the correct paper before using it.

For each baseline:
1. Get it running on its own example/demo data first — this proves your environment is set up correctly before you touch our data.
2. Swap in our DAVIS/KIBA splits from `data/splits/`.
3. Record accuracy metrics on all four splits (warm, cold-drug, cold-target, cold-pair).

Save the final results table as `results/baseline_comparison.csv` (three models × four splits × two datasets).

See `docs/01_GUIDE_124AD0008.md` Step 5 for the full walkthrough.

> Note: these baseline repos are excluded from git via `.gitignore` (each one has its own history and license) — only this README is tracked. Everyone regenerates them locally with the clone commands above.
