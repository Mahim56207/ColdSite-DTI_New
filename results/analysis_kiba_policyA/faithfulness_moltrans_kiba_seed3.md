# Faithfulness — moltrans_kiba_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.8823 | 0.8098 | **0.0724** | 3.2841 | 3.0455 | 0.9092 | 200 | yes |
| Cold-Drug | 0.6400 | 0.6494 | **-0.0094** | 2.5602 | 2.5388 | 0.6141 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
