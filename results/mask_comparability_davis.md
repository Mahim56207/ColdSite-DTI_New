# Are the two masking arms the same size of intervention? — moltrans

Masking the top-10 attended residues versus 10 random ones, measured as the fraction of the model's protein tokens that change (`src/evaluation/mask_comparability.py`, 20 DAVIS proteins).

- explanation's residues: **47.9%** of tokens change
- random residues: **95.0%**
- the random arm is therefore **2.0x** the intervention, not the same one

- a token-matched control lands at **61.6%** (median 55 draws), which is the comparison the faithfulness delta needs

| protein | tokens | attended | random | ratio |
|---|---|---|---|---|
| RIPK5 | 374 | 0.8% | 99.2% | 123.7x |
| FLT3(D835H) | 410 | 1.2% | 92.5% | 75.9x |
| ARK5 | 255 | 1.6% | 102.1% | 65.1x |
| PLK4 | 386 | 3.6% | 88.3% | 24.3x |
| GAK | 382 | 19.6% | 100.3% | 5.1x |
| ERK3 | 284 | 26.1% | 88.0% | 3.4x |
| IKK-alpha | 305 | 39.3% | 101.1% | 2.6x |
| PIK3CA(E545A) | 405 | 42.0% | 86.6% | 2.1x |
| NEK9 | 382 | 44.5% | 91.1% | 2.0x |
| MAPKAPK5 | 188 | 55.9% | 101.6% | 1.8x |
| RIPK4 | 298 | 54.4% | 97.5% | 1.8x |
| RSK3(KinDom.2-C-terminal) | 287 | 54.7% | 93.6% | 1.7x |
| KIT(V559D-V654A) | 391 | 56.5% | 88.7% | 1.6x |
| NEK5 | 285 | 63.2% | 94.2% | 1.5x |
| MKNK1 | 182 | 65.4% | 96.9% | 1.5x |
| AXL | 347 | 60.8% | 87.8% | 1.4x |
| IRAK1 | 277 | 73.6% | 95.4% | 1.3x |
| PIP5K2C | 165 | 96.4% | 100.0% | 1.0x |
| CTK | 199 | 97.5% | 95.8% | 1.0x |
| JNK1 | 169 | 101.2% | 98.8% | 1.0x |

**What this means for the audit.** A faithfulness delta computed against the scattered-random arm is not evidence about the explanation for a sub-word model: the arms differ in how much of the input they actually change. MolTrans's negative deltas are reported as this artefact, not as an anti-faithful explanation. Models that read one token per residue are unaffected, which `residue_level_models_are_unaffected` checks.
