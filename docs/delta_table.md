# Delta table — prior work vs this paper (T19, 2026-09-25)

Scope rule (plan T19): each "what it did" cell is drawn **only** from the work's verified abstract, saved in
`paper/citation_verification/<file>.json`. No full text was read, so nothing below characterises methods, numbers or
limitations the abstract does not state. The "what this paper adds" column describes this manuscript
(`paper/v2/introduction.md`, `paper/v2/methods.md`) and is limited to what the prior work's abstract lets us contrast.

| prior work (record) | what it did, per its abstract | what this paper adds |
|---|---|---|
| MONN 2020, *Cell Systems* (`Li2020_MONN.json`; abstract from the bioRxiv preprint) | Built a benchmark of inter-molecular non-covalent interactions for >10,000 compound–protein pairs and used it to evaluate the interpretability of neural attention in existing affinity models; reports that attention-based approaches did not give satisfactory interpretability **without extra supervision**; proposes MONN, which predicts interactions with supervision. | Asks not whether attention matches binding residues once, but whether the verdict is **stable across training seeds** of the same recipe, and reads each hit rate against chance, ceiling, a uniform map and a planted positive control with family-wise Holm correction; scored at four levels of distribution shift on kinase splits, with a faithfulness (masking) arm. |
| ICAN 2022, *PLOS ONE* (`Kurata2022_ICAN.json`) | Cross-attention DTI model on DAVIS (SMILES + sequence); reports that some weighted sites in the cross-attention matrix represent experimental binding sites "with statistical significance". | Re-examines this kind of single-model significance claim across three seeds, several readouts of the same attention, and cold-start splits; reports how often seeds disagree on the verdict. ICAN itself is not retrained here. |
| InteractBind 2026, arXiv 2605.24045 (`InteractBind_2605.24045.json`) — **preprint** | ~100k protein–ligand pairs with residue/atom interaction maps over six non-covalent interaction types; evaluates eight sequence-based and interaction-aware models on binding prediction and binding-site localisation, with affinity and protein-similarity-controlled splits; finds limited localisation despite strong binding prediction. | Complementary rather than overlapping in its abstract's terms: this paper contributes seed-to-seed verdict stability, a calibrated battery (chance, ceiling, uniform floor, positive control, positional and residue-identity nulls, Holm), readout dependence, and quantified DAVIS sequence leakage, on the models' own published recipes. |
| ISAAC 2026, arXiv 2605.02962 (`ISAAC_2605.02962.json`) — **preprint** | Post-hoc intervention framework (matched mechanistic vs spurious perturbations) applied to three sequence-based DTI architectures on DAVIS; reports reasoning-score differences across models of similar AUROC, stable across training and intervention seeds. | Different object: ISAAC scores sensitivity to structural interventions; this paper scores whether attention maps localise annotated sites and whether that verdict survives a change of seed. The two agree in treating accuracy as insufficient evidence of mechanism. |
| Vefghi 2026, arXiv 2606.14245 (`Vefghi2026_2606.14245.json`) — **preprint** | Interpretability audit of one model (BridgeDPI) on Gao, Human and C. elegans with five gradient-based attribution methods plus occlusion and cross-method consensus; states explicitly that the analyses do not substitute for structural or experimental ground truth. | Scores explanations **against** structural ground truth (UniProt sites, the KLIFS 85-residue pocket), on kinase benchmarks, for four attention models, with integrated gradients as a confirmatory comparator. |
| DAVIS-complete 2025, arXiv 2512.00708 (`DAVIScomplete_2512.00708.json`) — **preprint** | Curates a modification-aware DAVIS with 4,032 kinase–ligand pairs involving substitutions, insertions, deletions and phosphorylation; reports that docking-free models overfit to wild-type proteins. | Precedent that standard DAVIS under-represents protein modifications. This paper **quantifies** the consequence for cold-start evaluation: mutant targets carry wild-type sequences, so some "unseen" targets are seen by sequence, and reports accuracy on the sequence-unseen subset. |
| Ong 2023, bioRxiv 2023.09.04.556234 (`bioRxiv2023_556234.json`) — **preprint** | A CNN like published kinase-affinity models performs comparably under random splits but deteriorates when all data for an inhibitor are held out together; attributes the random-split performance to information leakage. | Precedent for split-induced leakage in kinase benchmarks (drug side). This paper measures a target-side leakage (sequence identity behind distinct identifiers) and its effect on accuracy and on explanation scoring. |

## Methodological references verified in T19 (not prior DTI work; no delta row)

Jain & Wallace 2019; Wiegreffe & Pinter 2019; ERASER 2020 (DeYoung et al.); Jacovi & Goldberg 2020; Adebayo et al. 2018;
DrugBAN 2023 (audited model); PSICHIC 2024 (candidate modern model, plan T14); KLIFS 2016 (Kooistra et al.). Records in
`paper/citation_verification/`.

## Not verified / not inserted

* **"The 2026 JCIM gradient-XAI DTI cold-split benchmark"** — the plan gives no title, author or DOI, and the one Crossref
  candidate found (doi:10.1021/acs.jcim.6c00037) did not match the description on "gradient" or "cold split".
  **Dropped entirely by the user's decision (2026-09-25):** not needed for the instructor draft, so it is neither cited nor
  listed in `paper/references.md`, and its candidate record was removed from `paper/citation_verification/`.
