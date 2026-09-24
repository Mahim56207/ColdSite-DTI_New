# Faithfulness delta with confidence intervals

Mean `comprehensiveness_delta` (attended masking minus the size-matched random control), pairs grouped by target, targets resampled (10,000 resamples, 95% percentile, `default_rng(0)`, each target carried in with all its seeds). `sign` is the original verdict (mean > 0); `CI` is the added one (lower bound > 0). Reported beside each other, neither replaces the other. MolTrans is in token space (size-matched arms), as in the paper.

| model | space | dataset | level | targets | pairs/seed | mean delta (pairs) | mean delta (targets) | 95% CI | sign | CI | max diff vs committed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coldsite_dti | residue | DAVIS | random | 162 | 200 | 1.1881 | 1.2100 | 1.0540–1.3776 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-drug | 200 | 200 | 1.1791 | 1.1791 | 1.0541–1.3057 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-target | 76 | 200 | 0.5166 | 0.5159 | 0.4420–0.5987 | yes | yes | 0.000000 |
| coldsite_dti | residue | DAVIS | cold-pair | 76 | 200 | 0.6652 | 0.6614 | 0.5757–0.7556 | yes | yes | 0.000000 |
| hyperattentiondti | residue | DAVIS | random | 162 | 200 | 0.1843 | 0.1892 | 0.1493–0.2326 | yes | yes | 0.000193 |
| hyperattentiondti | residue | DAVIS | cold-drug | 200 | 200 | 0.1135 | 0.1135 | 0.0870–0.1405 | yes | yes | 0.002574 |
| hyperattentiondti | residue | DAVIS | cold-target | 76 | 200 | 0.0633 | 0.0647 | 0.0462–0.0844 | yes | yes | 0.000073 |
| hyperattentiondti | residue | DAVIS | cold-pair | 76 | 200 | 0.0557 | 0.0542 | 0.0363–0.0727 | yes | yes | 0.000000 |
| hyperattentiondti | residue | KIBA | random | 114 | 200 | 0.1318 | 0.1348 | 0.1049–0.1662 | yes | yes | 0.000000 |
| hyperattentiondti | residue | KIBA | cold-drug | 112 | 200 | 0.0952 | 0.0874 | 0.0668–0.1096 | yes | yes | 0.000000 |
| moltrans | token | DAVIS | random | 200 | 200 | 0.4612 | 0.4612 | 0.4093–0.5157 | yes | yes | 0.017508 |
| moltrans | token | DAVIS | cold-drug | 200 | 200 | 0.3556 | 0.3556 | 0.3163–0.3962 | yes | yes | 0.012299 |
| moltrans | token | DAVIS | cold-target | 75 | 75 | 0.3360 | 0.3360 | 0.2737–0.4102 | yes | yes | 0.030428 |
| moltrans | token | DAVIS | cold-pair | 76 | 76 | 0.2584 | 0.2584 | 0.2123–0.3071 | yes | yes | 0.040319 |
| moltrans | token | KIBA | random | 200 | 200 | 0.3921 | 0.3921 | 0.3465–0.4408 | yes | yes | 0.022719 |
| moltrans | token | KIBA | cold-drug | 200 | 200 | 0.3025 | 0.3025 | 0.2676–0.3388 | yes | yes | 0.019449 |
