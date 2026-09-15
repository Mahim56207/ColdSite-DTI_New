# Faithfulness — moltrans_kiba_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.9615 | 1.1120 | **-0.1505** | 2.3394 | 2.3467 | 1.0214 | 200 | **no** |
| Cold-Drug | 0.9304 | 0.9660 | **-0.0355** | 1.5798 | 1.5749 | 0.9432 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
