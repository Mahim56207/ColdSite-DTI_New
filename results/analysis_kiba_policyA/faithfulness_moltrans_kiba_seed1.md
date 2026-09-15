# Faithfulness — moltrans_kiba_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.8786 | 0.9362 | **-0.0576** | 1.8246 | 1.9445 | 0.9593 | 200 | **no** |
| Cold-Drug | 0.7505 | 0.7215 | **0.0290** | 1.3986 | 1.4936 | 0.8037 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
