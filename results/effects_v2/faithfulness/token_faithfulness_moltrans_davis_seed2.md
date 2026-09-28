# Faithfulness in token space — moltrans, davis, seed 2

The intervention is a token, which is what this model reads, and the control removes the same number of tokens as the explanation does -- so both arms change an identical amount of the input (`src/evaluation/token_faithfulness.py`). The residue-space test could not do that: masking its top-10 residues changes 48% of its tokens where 10 random residues change 95%.

| Level | comp. | random control | **delta** | suff. | suff. random | tokens | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| random | 0.6376 | 0.1589 | **+0.4786** | 1.0037 | 1.3518 | 256 | 200 | yes |
| cold-drug | 0.4230 | 0.1920 | **+0.2310** | 1.3193 | 1.3432 | 250 | 200 | yes |
| cold-target | 0.5563 | 0.2298 | **+0.3266** | 0.6687 | 0.7548 | 260 | 75 | yes |
| cold-pair | 0.4099 | 0.2294 | **+0.1805** | 0.7899 | 0.8123 | 293 | 76 | yes |

`delta` is the only column that is a result. A negative delta here means the explanation's tokens matter less than an equal number of arbitrary ones -- which, unlike the residue-space version, is a statement about the attention rather than about the intervention.
