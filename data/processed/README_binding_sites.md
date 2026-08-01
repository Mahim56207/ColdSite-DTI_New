# Binding Site Ground Truth

This directory contains the ground truth binding and active site annotations for the ColdSite-DTI evaluation.

## Files
- `binding_sites_ground_truth.json`: The core deliverable mapping UniProt Accessions to residue locations.

## Format
The JSON contains two main keys:
1. `metadata`: Contains dataset provenance and the lookup dictionary used to map DAVIS gene names to UniProt accessions.
2. `binding_sites`: A dictionary where keys are UniProt Accession IDs, and values are lists of location objects (extracted directly from the UniProt Features API for 'Binding site' and 'Active site').

Hand this JSON file to the evaluation team (124AD0067) for their spatial overlap audits.
