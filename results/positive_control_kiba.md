# Positive control — kiba

Explanations of known quality, scored by the audit's own functions (k = 10, 1000 permutation trials, one test row per protein, 1000-residue window). See `src/evaluation/positive_control.py`.

## Plausibility: precision@k by dose

| level | n | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 | ceiling | chance | smallest detectable dose |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 211 | 0.023 | 0.046* | 0.095* | 0.163* | 0.266* | 0.607* | 0.995* | 0.995 | 0.023 | 0.02 |
| cold_drug | 212 | 0.022 | 0.039* | 0.084* | 0.147* | 0.276* | 0.631* | 0.995* | 0.995 | 0.023 | 0.02 |
| cold_target | 42 | 0.014 | 0.038* | 0.093* | 0.162* | 0.331* | 0.633* | 0.998* | 0.998 | 0.022 | 0.02 |
| cold_pair | 41 | 0.029 | 0.056* | 0.095* | 0.154* | 0.349* | 0.720* | 0.985* | 0.985 | 0.027 | 0.02 |

`*` = significantly above chance (split-level permutation test, p < 0.05).

## Faithfulness on a planted model: comprehensiveness delta by dose

| level | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 |
|---|---|---|---|---|---|---|---|
| random | 0.0488 | 0.2689 | 0.6529 | 1.2481 | 2.5286 | 6.4191 | 10.3199 |
| cold_drug | -0.0275 | 0.3123 | 0.5960 | 1.2719 | 2.7987 | 6.4581 | 10.3218 |
| cold_target | -0.0548 | 0.2392 | 0.4809 | 1.0808 | 2.2009 | 6.0954 | 10.4955 |
| cold_pair | 0.0260 | 0.3260 | 0.9091 | 1.4231 | 2.4926 | 7.1608 | 10.1085 |

The planted model's prediction depends only on the annotated sites, so the oracle (dose 1) must be load-bearing and dose 0 should sit near zero.

## Checks

- **PASS** random: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** random: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** random: oracle p = 0.000999 (must be < 0.05)
- **PASS** random: planted-model oracle comprehensiveness delta = +10.3199 (must be > 0)
- **PASS** cold_drug: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_drug: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_drug: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_drug: planted-model oracle comprehensiveness delta = +10.3218 (must be > 0)
- **PASS** cold_target: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_target: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_target: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_target: planted-model oracle comprehensiveness delta = +10.4955 (must be > 0)
- **PASS** cold_pair: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_pair: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_pair: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_pair: planted-model oracle comprehensiveness delta = +10.1085 (must be > 0)
