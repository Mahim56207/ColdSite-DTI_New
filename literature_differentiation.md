# Literature Review and Differentiation

This document outlines the core differentiation between ColdSite-DTI and existing literature, serving as the backbone for the Related Work section.

### 1. DMFF-DTA (2025)
DMFF-DTA evaluates a model's internal attention weights against known biological binding sites using rigorous statistical testing frameworks. However, its entire analytical pipeline is restricted to standard, non-cold-start data splits. ColdSite-DTI extends this validation methodology directly into strict cold-start environments—spanning drugs, targets, and pairs—to measure how explanation faithfulness holds up when models face entirely unfamiliar biological data.

### 2. EviDTI (2025)
EviDTI calculates a "binding site hit ratio" to measure attention accuracy while separately examining model performance on cold-start prediction tasks. Crucially, these two evaluations are treated as independent experiments rather than being connected analytically. ColdSite-DTI bridges this gap by quantifying explanation accuracy as a direct, continuous function of cold-start split difficulty, producing a unified curve that ties generalization performance straight to interpretive fidelity.

### 3. ColdDTI (2025)
ColdDTI achieves strong predictive accuracy under difficult cold-start conditions by leveraging multi-level protein structural representations. Despite its robust predictive performance, it does not assess the quality or biological validity of the model's internal explanations. While ColdDTI focuses strictly on predicting outcomes in cold-start scenarios, ColdSite-DTI investigates whether the model arrives at those predictions for the correct biological reasons by tracking attention precision across the same cold-start spectrum.

### 4. CS-DTA (2026)
CS-DTA addresses both predictive generalization and model interpretability, making it the closest methodological precedent in current literature by claiming capabilities in both domains. ColdSite-DTI differentiates itself by explicitly charting explanation precision as a granular, quantified function of varying cold-start severity levels (warm, cold-drug, cold-target, and cold-pair). Rather than offering a generalized claim of interpretability, ColdSite-DTI provides a precise statistical trajectory showing exactly how explanation accuracy degrades or survives as generalization gets harder.

### 5. GPS-DTI (2025)
GPS-DTI achieves powerful cold-start performance by combining advanced protein language models like ESM-2 with attention mechanisms. However, it evaluates predictive capability without measuring whether explanation fidelity degrades as split difficulty increases. ColdSite-DTI builds upon modern embedding architectures by directly auditing the model's attention against biological binding sites across a complete ladder of cold-start difficulties, ensuring that high predictive performance actually correlates with correct biological focus.