# Faithfulness in token space — moltrans, davis, seed 3

The intervention is a token, which is what this model reads, and the control removes the same number of tokens as the explanation does -- so both arms change an identical amount of the input (`src/evaluation/token_faithfulness.py`). The residue-space test could not do that: masking its top-10 residues changes 48% of its tokens where 10 random residues change 95%.

| Level | comp. | random control | **delta** | suff. | suff. random | tokens | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| random | 0.9327 | 0.1702 | **+0.7625** | 1.3714 | 2.0120 | 256 | 200 | yes |
| cold-drug | 0.8162 | 0.1684 | **+0.6478** | 1.1522 | 1.5358 | 250 | 200 | yes |
| cold-target | 0.7035 | 0.1699 | **+0.5335** | 0.6308 | 0.9844 | 260 | 75 | yes |
| cold-pair | 0.6701 | 0.1677 | **+0.5024** | 1.7222 | 1.8493 | 293 | 76 | yes |

`delta` is the only column that is a result. A negative delta here means the explanation's tokens matter less than an equal number of arbitrary ones -- which, unlike the residue-space version, is a statement about the attention rather than about the intervention.
