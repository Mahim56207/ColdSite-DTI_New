# Sequence audit — kiba

What the model reads for each target. Produced by `src/data/sequence_audit.py`; changes nothing, reports only.

**229 distinct sequences for 229 targets.**

## 1. Variants identical to their wild type

0 variant targets: **0 carry exactly the wild-type sequence**, 0 differ from it, 0 have no wild-type entry to compare.


## 2. Test targets unseen by name but seen by sequence

| level | part | targets | unseen by name | identical sequence in training | rows affected |
|---|---|---|---|---|---|
| random | test | 228 | 0 | 0 | 0 of 23651 (0.0%) |
| random | valid | 226 | 0 | 0 | 0 of 11825 (0.0%) |
| cold_drug | test | 229 | 0 | 0 | 0 of 22374 (0.0%) |
| cold_drug | valid | 226 | 0 | 0 | 0 of 12073 (0.0%) |
| cold_target | test | 45 | 45 | 0 | 0 of 22101 (0.0%) |
| cold_target | valid | 22 | 22 | 0 | 0 of 10701 (0.0%) |
| cold_pair | test | 45 | 45 | 0 | 0 of 4375 (0.0%) |
| cold_pair | valid | 22 | 22 | 0 | 0 of 1334 (0.0%) |

## 3. Sequences without the kinase pocket

Targets whose sequence holds fewer than 40 of the 85 KLIFS pocket residues (`data/kiba_klifs_pocket_report.json`): **0**

