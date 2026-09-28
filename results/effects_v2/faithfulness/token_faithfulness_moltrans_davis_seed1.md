# Faithfulness in token space — moltrans, davis, seed 1

The intervention is a token, which is what this model reads, and the control removes the same number of tokens as the explanation does -- so both arms change an identical amount of the input (`src/evaluation/token_faithfulness.py`). The residue-space test could not do that: masking its top-10 residues changes 48% of its tokens where 10 random residues change 95%.

| Level | comp. | random control | **delta** | suff. | suff. random | tokens | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| random | 0.3339 | 0.1913 | **+0.1426** | 1.2861 | 1.2345 | 256 | 200 | yes |
| cold-drug | 0.4492 | 0.2612 | **+0.1880** | 1.0819 | 1.1309 | 250 | 200 | yes |
| cold-target | 0.3365 | 0.1885 | **+0.1480** | 0.4698 | 0.5728 | 260 | 75 | yes |
| cold-pair | 0.2725 | 0.1802 | **+0.0923** | 0.5311 | 0.4584 | 293 | 76 | yes |

`delta` is the only column that is a result. A negative delta here means the explanation's tokens matter less than an equal number of arbitrary ones -- which, unlike the residue-space version, is a statement about the attention rather than about the intervention.
