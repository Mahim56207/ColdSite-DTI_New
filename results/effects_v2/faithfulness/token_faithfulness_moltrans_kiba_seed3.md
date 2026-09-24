# Faithfulness in token space — moltrans, kiba, seed 3

The intervention is a token, which is what this model reads, and the control removes the same number of tokens as the explanation does -- so both arms change an identical amount of the input (`src/evaluation/token_faithfulness.py`). The residue-space test could not do that: masking its top-10 residues changes 48% of its tokens where 10 random residues change 95%.

| Level | comp. | random control | **delta** | suff. | suff. random | tokens | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| random | 0.7635 | 0.1409 | **+0.6226** | 1.8228 | 1.8934 | 254 | 200 | yes |
| cold-drug | 0.3507 | 0.1073 | **+0.2434** | 1.5093 | 1.3422 | 236 | 200 | yes |
| cold-target | — | — | — | — | — | — | 0 | — |
| cold-pair | — | — | — | — | — | — | 0 | — |

`delta` is the only column that is a result. A negative delta here means the explanation's tokens matter less than an equal number of arbitrary ones -- which, unlike the residue-space version, is a statement about the attention rather than about the intervention.
