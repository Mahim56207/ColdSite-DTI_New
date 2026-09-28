# Faithfulness delta with confidence intervals

Mean `comprehensiveness_delta` (attended masking minus the size-matched random control), pairs grouped by target within each seed. TWO-WAY bootstrap (`--resample seeds_and_targets`), 10,000 resamples, 95% percentile intervals, `default_rng(0)`: each resample draws the targets with replacement and, independently, the seeds with replacement from the cell's seeds, and averages over the drawn (target, seed) grid, so an interval carries seed-to-seed variance as well as target sampling. Point values are identical to the target-only tables of `results/effects_v2/` (amendment section 5); only the intervals differ. `sign` is the original verdict (mean > 0); `CI` is the added one (lower bound > 0). MolTrans is in token space (size-matched arms), as in the paper.

| model | space | dataset | level | targets | pairs/seed | mean delta (pairs) | mean delta (targets) | 95% CI | sign | CI | max diff vs committed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coldsite_dti | residue | DAVIS | random | 162 | 200 | 1.1881 | 1.2100 | 0.6834–1.9189 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-drug | 200 | 200 | 1.1791 | 1.1791 | 0.7609–1.7422 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-target | 76 | 200 | 0.5166 | 0.5159 | 0.1833–0.8754 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-pair | 76 | 200 | 0.6652 | 0.6614 | 0.4299–0.9412 | yes | yes | 0.000000 |
| hyperattentiondti | residue | DAVIS | random | 162 | 200 | 0.1843 | 0.1892 | 0.1228–0.2769 | yes | yes | 0.000193 |
| hyperattentiondti | residue | DAVIS | cold-drug | 200 | 200 | 0.1135 | 0.1135 | 0.0566–0.1688 | yes | yes | 0.002574 |
| hyperattentiondti | residue | DAVIS | cold-target | 76 | 200 | 0.0633 | 0.0647 | 0.0185–0.1221 | yes | yes | 0.000073 |
| hyperattentiondti | residue | DAVIS | cold-pair | 76 | 200 | 0.0557 | 0.0542 | 0.0297–0.0804 | yes | yes | 0.000000 |
| hyperattentiondti | residue | KIBA | random | 114 | 200 | 0.1318 | 0.1348 | 0.0693–0.2235 | yes | yes | 0.000000 |
| hyperattentiondti | residue | KIBA | cold-drug | 112 | 200 | 0.0952 | 0.0874 | 0.0304–0.1439 | yes | yes | 0.000000 |
| moltrans | token | DAVIS | random | 200 | 200 | 0.4612 | 0.4612 | 0.1553–0.7506 | yes | yes | 0.017508 |
| moltrans | token | DAVIS | cold-drug | 200 | 200 | 0.3556 | 0.3556 | 0.1748–0.6308 | yes | yes | 0.012299 |
| moltrans | token | DAVIS | cold-target | 75 | 75 | 0.3360 | 0.3360 | 0.1566–0.5501 | yes | yes | 0.030428 |
| moltrans | token | DAVIS | cold-pair | 76 | 76 | 0.2584 | 0.2584 | 0.0899–0.4872 | yes | yes | 0.040319 |
| moltrans | token | KIBA | random | 200 | 200 | 0.3921 | 0.3921 | 0.1293–0.6227 | yes | yes | 0.022719 |
| moltrans | token | KIBA | cold-drug | 200 | 200 | 0.3025 | 0.3025 | 0.2277–0.4022 | yes | yes | 0.019449 |
