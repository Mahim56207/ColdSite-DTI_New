# Venue specification — written 2026-10-17

**Chosen target: ISMB 2027 Proceedings track (published as a special issue of *Bioinformatics*), fall-back: *Bioinformatics* Original Paper, then *Briefings in Bioinformatics*.**
Why this one: `docs/00_MASTER_PLAN_V2.md` §4 already names ISMB / Bioinformatics / Briefings and rules out chasing an A* ML conference;
ISMB is the one *conference* among them, and its proceedings are held to the *Bioinformatics* journal's format. An audit paper with a
benchmarking contribution fits that venue; it is not a new-model paper and should not be pitched as one.

## Verified (fetched from the official page on 2026-10-17)

| Item | Value | Source |
|---|---|---|
| Original Paper length | "Up to 7 pages; this is approx. 5,000 words. excluding figures" | https://academic.oup.com/bioinformatics/pages/author-guidelines |
| Review model | single-anonymised (authors are visible to reviewers), typically 3 reviewers | same |
| Code | free, at a stable URL (e.g. GitHub), no request needed for non-commercial use; availability stated in the manuscript | same |
| Archiving | software version and test data archived in a dedicated repository (Zenodo, Figshare, Software Heritage, …); its URL goes "in the availability section of the abstract" | same |
| Data | all data behind the conclusions public in stable repositories; a data availability statement is required | same |
| Supplement | submitted for review as a separate file, all extra text, figures and tables in one single file; cannot be replaced after acceptance | same |
| ISMB 2026 Proceedings | call opened 11 Nov 2025, deadline **20 Jan 2026 23:59 AoE, no extensions**; "new, unpublished work"; accepted papers form an open-access special issue of *Bioinformatics* | https://www.iscb.org/about-iscb/society-communications/announcements/november-11-2025-call-for-papers-ismb-2026-proceedings-now-open |

## NOT verified — do not state these as facts in any document

- The **ISMB 2027** call, deadline, page limit and review model. No 2027 call was found on 2026-10-17. The 2026 timetable (call in November, deadline the following 20 January)
  is a *pattern*, not a commitment; check iscb.org in November.
- ISMB's own proceedings page limit. Older ISMB calls (2002, 2013) said 9 pages with the journal template; those are old and not to be relied on.
- The abstract's structure and word limit, the keyword rule, and the reference style: the author-guidelines page did not state them. The *Bioinformatics* convention
  (a structured abstract with Motivation / Results / Availability and implementation / Contact / Supplementary information, about 250 words) is from memory and must be
  confirmed against the template on the journal's site before submission.
- Whether the ISMB proceedings track requires the *Bioinformatics* LaTeX/Word template, and its preprint policy.

## Consequences for this paper

1. **Length.** The main text must fit ≈ 5,000 words excluding figures. `paper/v2/` is ≈ 10,000 words of main text (abstract 1,035, introduction 2,043, methods 2,360,
   results 3,054, discussion 1,502), so the venue version is a **condensation by about half**, with the rest moved to the single supplement. Nothing is cut that a claim depends on;
   it moves to the supplement and the main text points to it.
2. **Single-anonymised.** No anonymisation is needed, but the in-house model's name rule still stands: the code name appears only in the naming note.
3. **Availability needs two things the repository does not yet have:** an archived release with a DOI (Zenodo) and a data statement. Neither exists (no Zenodo mention, no `CITATION.cff` in the repo).
   This is a to-do for the author, not something to write into the paper as if done.
4. **Novelty to be unpublished.** The paper must not be posted elsewhere as a finished article before submission; a preprint policy for ISMB is unverified, so check before posting anything.
5. **Pending science.** Seeds 4 and 5 (Wave A) have not been run. The paper as it stands reports three seeds; if Wave A lands before submission, every seed-count statement and the 11-of-22 headline must be recomputed
   (`results/certification/key_numbers.csv`), not edited by hand.

## Working rules for the writing (no hallucination)

* Every digit in the main text carries a source tag and passes `python scripts/check_number_provenance.py`. No number is typed from memory.
* No citation is added unless it is in `paper/references.md` and its verification record is in `paper/citation_verification/`, or is newly checked against the source and recorded.
* A sentence about what the models or data *are* comes from the repository's own files; a sentence about what other papers *claim* comes from a verified citation, or it is omitted.
* Claims keep their calibrated wording: "1 of 20 DAVIS cells", "11 of 22 cells disagree", the Holm = median-over-seeds definition, the leaked-row +0.116 first, the readout dependence as a **measurement** result, not a proof.
* Known caveat to carry into the limitations: the same-seed rerun noise (canary: AUROC 0.8868 vs 0.8494 on identical initial weights) means part of the "seed" variation is run-to-run variation; the paper's own wording must say so. It is recorded in `docs/canary_attempts/README.md`.
* Currently failing provenance tags: 14, all pointing at `src/cloud/recipes.py` line numbers that moved when DrugBAN's precision rule was edited (uncommitted). Re-point them once that edit is final.
