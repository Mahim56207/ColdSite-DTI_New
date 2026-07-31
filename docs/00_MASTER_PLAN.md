# ColdSite-DTI — Master Project Plan

**Team:** 124AD0008 · 124AD0015 · 124AD0067
**Course:** Bioinformatics — B.Tech AI & Data Science, 5th Semester
**Supervisor:** Dr. Chandra Mohan Dasari
**Working window:** August 2026 – November 2026 (flexible up to end of November)

---

## 0. How to use these 4 files

This file is the map of the whole project. Read it fully, as a team, before anyone writes a line of code. It explains **what** we are building, **why** it matters, and **how** the three of you fit together.

After this, each person opens **only their own file**:

- `01_GUIDE_124AD0008.md`
- `02_GUIDE_124AD0015.md`
- `03_GUIDE_124AD0067.md`

Every guide has the same shape: **Part 1 is solo work**, something you can start today without waiting on anyone else. **Part 2 is joint work**, which starts only once all three of you have finished Part 1. Nobody is blocked at the start — that's on purpose.

---

## 1. The problem we are solving, explained simply

There are AI models that look at a drug and a protein and predict: "will this drug stick to this protein or not?" This is called **Drug–Target Interaction (DTI) prediction**. It matters because testing this in a real wet lab is slow and expensive — a computer prediction lets scientists narrow thousands of candidate drugs down to a shortlist worth actually testing.

Some of these models are "explainable" — alongside the yes/no answer, they also highlight *which part* of the protein they think the drug is sticking to. This is usually done with something called **attention** (explained in the glossary below). The model basically says: "I predict this drug binds, and here is the exact spot on the protein I think it binds to."

Here is the catch, and this is the whole point of our project:

> Imagine a doctor who is brilliant at diagnosing diseases they've seen a thousand times, and who can point to the exact symptoms that led to each diagnosis. Now give that doctor a disease they have never seen before. Do they still diagnose it correctly? And just as important — is their explanation this time still trustworthy, or are they just telling a plausible-sounding story after the fact?

That is exactly the question nobody has properly answered for DTI models. Everybody tests whether the *yes/no prediction* stays accurate when the model sees a drug or protein it has never encountered before ("cold-start"). Separately, people check whether the model's explanation *usually* points at real binding locations. **Nobody has put these two things on the same graph and asked: does the explanation stay trustworthy specifically when the model is facing something totally new — which is the exact moment a scientist would need to trust it the most?**

That gap is our paper.

---

## 2. Why this matters (in plain terms)

If a pharmaceutical scientist wants to find a new drug for a *new* disease target — say, a newly discovered virus — they cannot rely on a model that only works well on drugs and proteins it already memorized during training. They need the model to work, and to explain itself honestly, on things it has genuinely never seen. If the model's explanation quietly stops being trustworthy exactly in that situation — and nobody warned them — that's a real, practical danger, because a scientist might trust a highlighted "binding spot" that is actually meaningless, and waste months chasing it.

We are going to measure this danger directly, put a number on it, and report it. That is a real, useful, publishable contribution — not just an incremental accuracy improvement.

---

## 3. What we are actually building: ColdSite-DTI

**In one sentence:** We build a drug–protein binding predictor, in the style of our professor's own published architecture, and instead of just reporting "is it accurate," we specifically measure whether its explanations point to real, known binding locations — and we measure this separately at every level of "how unfamiliar is this drug/protein to the model," from fully familiar to completely brand new.

### 3.1 The model

The model reads two things:
1. **A drug**, written as a short text code called **SMILES** (a compact way of writing a molecule's structure as text — the glossary below explains this).
2. **A protein**, written as its **amino acid sequence** (a long string of letters, each letter representing one building block of the protein).

It processes each one through its own encoder (a piece of neural network that turns raw text into a meaningful set of numbers), combines the two using **attention** (a mechanism that lets the model decide which parts of the protein matter most for this particular drug), and outputs a prediction: does this drug bind to this protein, and how strongly?

We are deliberately copying the architecture style Dr. Dasari already published in his own transcription-factor binding-site paper — parallel CNN + BiLSTM layers followed by a multi-head attention layer. This is not accidental. It means our paper is a direct, natural continuation of his own published methodology, moved into a new problem (drug–protein binding instead of DNA–protein binding). That is a very strong "why this team" story for a reviewer, and for him.

### 3.2 The measurement (this is the actual contribution)

For every prediction, the model's attention tells us which protein positions it focused on. We check: **do those highlighted positions match the protein's real, known binding site** (the actual physical pocket where drugs attach, according to real experimental databases)? We measure this as a precise number (called **precision@k** — explained in the glossary).

We do this measurement **four times**, at four levels of difficulty:

| Level | What the model has seen before | What this simulates |
|---|---|---|
| 1. Warm / random split | Drug: yes. Protein: yes. | The easy, unrealistic case most papers report |
| 2. Cold-drug | Drug: **no, brand new.** Protein: yes. | Testing a new candidate drug on a well-studied target |
| 3. Cold-target | Drug: yes. Protein: **no, brand new.** | Testing a known drug against a newly discovered target |
| 4. Cold-pair | Drug: **no.** Protein: **no.** | The realistic, hardest case — genuinely new science |

We plot how the explanation-accuracy number changes across these four levels, right next to how the plain yes/no-accuracy number changes. **That plot is the headline result of the paper.**

### 3.3 The real-world case study

We run the same test specifically on well-known antiviral drug targets: **SARS-CoV-2 main protease (Mpro) and RNA-dependent RNA polymerase (RdRp), HIV protease and reverse transcriptase, and influenza neuraminidase.** These all have extremely well-documented binding sites, connecting our results to something concrete and connecting back to Dr. Dasari's own past work on viral genome prediction.

---

## 4. Why this is genuinely new (so you can explain it confidently)

We checked this carefully before committing to it. Two things are true at the same time:

1. Checking "does attention line up with known binding sites" has been done before, in several 2024–2025 papers.
2. Testing "does accuracy survive cold-start" has also been done before, in several 2025–2026 papers.

**But nobody has explicitly combined them** — measuring the *explanation's* accuracy as a function of *how cold* the test case is, and reporting that curve as the headline finding. That specific, focused question is what makes this our paper and not a repeat of someone else's.

**Important team responsibility:** because this exact corner of the field is moving fast, every team member must read the five related papers listed in Section 9 before we finalize our results, and each of us should be able to say in one sentence how ColdSite-DTI differs from each one. This protects our novelty claim.

---

## 5. The team and how the work splits

We are splitting the project into three roughly equal, genuinely independent tracks. Nobody has to wait for anyone else to begin — every solo part can start the same day.

| Person | Track | Owns |
|---|---|---|
| **124AD0008** | **A — Data, Cold-Start Splits & Baselines** | Getting and cleaning the datasets, building the four difficulty splits, collecting real binding-site ground truth, reproducing three comparison models. Also the point of contact for the team and keeper of the shared tracker. |
| **124AD0015** | **B — Core Model (ColdSite-DTI itself)** | Designing and building the actual drug+protein model, the attention mechanism, and the training loop. |
| **124AD0067** | **C — Explanation Testing, Case Study & Literature** | Building the precision@k measurement code, running the antiviral case study, and writing the literature-differentiation notes. |

This is fair: each track is a full, substantial piece of real research work, and each is buildable without waiting on the others (details of exactly how are in each person's own guide).

---

## 6. Timeline

We have from now (end of July 2026) until end of November 2026 — about 17 weeks. Here is the shape of it:

### August — Foundation Month
Everyone learns the concepts in their own guide, sets up their tools, and starts their solo work. By end of August, each person should have a first working version of their piece (even if rough).

### September — Build Month
Solo work continues and gets finished. Around mid-to-late September, the team starts **joint work**: plugging the three pieces together into one working pipeline.

### October — Experiment Month
Run the full set of experiments: all four splits, both metrics, on DAVIS and KIBA, plus the antiviral case study. This is when the actual results — the numbers and plots for the paper — get produced. Leave time to re-run things; first results are rarely final.

### November — Writing Month
Analyze results, add any extra stretch-goal experiments if time allows, write the paper together, revise it, and prepare it for submission. Aim to have a full draft done by mid-November, leaving the last two weeks for polishing, not first-drafting.

**Rule of thumb:** if by 15 September any track isn't producing real output, that is the moment to ask for help from the other two — don't wait until October to raise a flag.

---

## 7. Glossary — key terms in plain English

- **SMILES** — a short text string that describes a molecule's structure. Example: water is `O`, and aspirin is written as a longer string of letters and symbols. It's like a compact sentence describing a 3D shape.
- **Amino acid sequence** — a protein written as a string of letters, each letter standing for one of the 20 building blocks (amino acids) that make up the protein chain.
- **Binding site (or active site)** — the specific "pocket" on a protein's surface where a drug molecule physically attaches, like a lock that a specific key fits into.
- **Attention (in a neural network)** — a mechanism that lets the model assign a higher "importance score" to some parts of the input than others when making its decision, similar to highlighting the important words in a sentence.
- **CNN (Convolutional Neural Network)** — a network that slides a small window across the input looking for repeating local patterns, useful for spotting short recurring motifs in a sequence.
- **BiLSTM (Bidirectional LSTM)** — a network that reads a sequence both forwards and backwards, so it understands each position using context from both directions.
- **Cold-start** — testing a model on a drug or protein it never saw during training, which is the realistic situation in real drug discovery.
- **Precision@k** — if you take the model's top-k highest-attention positions, what fraction of them are genuinely correct (real binding-site positions)? A simple, honest accuracy number for an explanation.
- **AUROC / AUPRC** — standard accuracy scores for a yes/no prediction task; higher is better, 0.5 is random guessing.
- **Baseline** — an existing, already-published model that we run on the same data, so we have a fair number to compare our model against.

---

## 8. Tools and datasets, at a glance

- **Data:** DAVIS and KIBA datasets (drug–protein binding data), accessed through a Python package called **PyTDC** (Therapeutics Data Commons) — `pip install PyTDC`. It has DAVIS and KIBA built in, and can build cold-start splits automatically.
- **Antiviral subset:** filtered from BindingDB, focused on SARS-CoV-2 Mpro/RdRp, HIV protease/RT, and influenza neuraminidase.
- **Binding-site ground truth:** scPDB (bioinfo-pharma.u-strasbg.fr/scPDB) and/or BioLiP, cross-checked against UniProt's own binding-site annotations (accessible through UniProt's free web API).
- **Baselines to reproduce:** DeepDTA, HyperAttentionDTI, MolTrans — all have public code online.
- **Core libraries:** Python, PyTorch, RDKit (for handling SMILES/molecules), pandas, scikit-learn, matplotlib.
- **Compute:** the college HPC machines are enough for this entire project — nothing here needs a huge model. Free Google Colab / Kaggle notebooks are a fine backup for anyone's individual training runs.

---

## 9. Required reading — the five papers we must be able to differentiate from

Every team member reads at least one of these and writes a short paragraph (in their own words) explaining how ColdSite-DTI is different. Divide these across the team however feels natural — a good default is each person takes one or two, prioritizing whichever relates most to their own track.

1. **DMFF-DTA** (2025) — checks attention against binding sites using statistical testing, but on standard (not cold-start) splits.
2. **EviDTI** (2025) — reports a "binding site hit ratio" for attention, and separately tests cold-start accuracy, but does not connect the two.
3. **ColdDTI** (2025) — strong cold-start accuracy results using multi-level protein structure, but does not test explanation quality.
4. **CS-DTA** (2026) — claims both generalization and interpretability, the closest existing paper to ours — read this one especially carefully.
5. **GPS-DTI** (2025) — strong cold-start results using ESM-2 and attention, without measuring explanation fidelity as a function of split difficulty.

---

## 10. What the finished paper will contain

- **Abstract & Introduction** — the problem, the gap, our contribution (written together, last).
- **Related Work** — built from Section 9's reading (mainly Track C, reviewed by everyone).
- **Methods** — the model architecture (Track B) and the data/splits/ground-truth setup (Track A).
- **Experiments & Results** — the four-level ladder table and plot, baseline comparisons, the antiviral case study (Track C leads the writing, using outputs from A and B).
- **Discussion** — what the findings mean for real drug discovery, honestly including any results that don't look flattering — that honesty is a strength, not a weakness, in front of reviewers.
- **Conclusion & Future Work** — written together.

---

## 11. Definition of done

The project is complete when we have:

- [ ] Cleaned DAVIS and KIBA data with all four splits (warm, cold-drug, cold-target, cold-pair) built and saved.
- [ ] Real binding-site ground truth collected and matched to our protein set.
- [ ] Three baseline models reproduced and evaluated on the same splits.
- [ ] ColdSite-DTI trained and evaluated on all four splits, on both datasets.
- [ ] The explanation-fidelity ladder (precision@k across all four splits) computed and plotted, alongside accuracy.
- [ ] The antiviral case study completed for all named targets.
- [ ] A written differentiation note against each of the five required-reading papers.
- [ ] A full paper draft with all sections above.
- [ ] A submission-ready version, formatted for our target venue.

---

## 12. If things go wrong (and they will, a little — that's normal research)

- **Behind schedule by mid-October?** Drop BindingDB robustness checks and the optional masking-based faithfulness layer first — the core DAVIS/KIBA ladder result is what matters most and must survive.
- **A dataset or database is hard to access?** Don't lose more than a day or two on it alone — flag it to the team immediately; there is almost always a backup source (see each guide's troubleshooting section).
- **Results look "boring" (explanation fidelity doesn't drop much under cold-start)?** That is still a valid, reportable, honest finding — do not force or cherry-pick a more dramatic result. Report what actually happens.
- **Not sure which venue to submit to?** Keep writing at full rigor regardless — a well-run version of this project is submittable to a workshop, a solid conference, or a good journal; we decide the exact venue closer to November once we know the strength of our results.

---

Next: each person opens their own guide and starts Part 1 today.
