# Governed BASiC/ASAP — private manuscript review snapshot

Code, historical slice-registry metadata and screened non-RIPAS aggregate outputs supporting research on governed hidden-failure slice discovery and targeted repair for emergency-department admission prediction.

**BASiC**: Bias-aware Auditable Slices in Clinical Pre-processing. **ASAP**: Auditable Slice Acceptance Protocol.

## Status and scope

Prepared on 16 October 2026 for a **private** repository owned by `nazirkaoje777-dotcom`. This is a partial review snapshot, not an unrestricted data release, a complete reproduction of every manuscript table, or a complete new reproduction. The corrected attribute-free comparison is a new experiment run, explicitly separated from stored historical results. No patient-level records or fitted models are included. Publication and reuse permissions remain subject to the relevant authors, institution and data custodians.

### Included

- `Code/`: analysis scripts, including the new `Code/attribute_free/` rerun. Path adaptations and unchanged statistical functions are documented in `Documentation/ATTRIBUTE_FREE_CORRECTION.md`.
- `basic_slice_prem/Notebooks/`: ten historical notebooks with outputs, attachments and execution metadata removed.
- `Registry/`: the historical 29-row slice-level registry, labelled to distinguish original qualitative judgments from the separate later clinician questionnaire.
- `Results/`: stored non-RIPAS aggregate tables, grouped by analysis stage. Mixed and negative findings are retained.
- `Documentation/`: parameters, provenance, sharing boundaries and scientific limitations.
- `RELEASE_MANIFEST.csv` and `tools/verify_release.py`: checksums and an automated disclosure/integrity screen.

### Intentionally withheld

- All patient records, row-level outcomes/predictions, profile identifiers, row indexes, membership matrices, trained models and runtime outputs.
- All RIPAS-derived result tables and mixed external-analysis manifests, pending explicit permission for third-party repository sharing. Code describing harmonization is included; ethics approval alone is not treated as redistribution permission.
- Encounter-level audit outputs from the corrected attribute-free rerun. Only screened code and aggregate results are included; the superseded comparison is labelled historical.
- Clinician responses, adjudications, identifying review information, supervisor comments, correspondence, manuscript drafts, approval letters/signatures and third-party paper PDFs.

See `Documentation/REVIEW_STATUS.md` and `Documentation/DATA_SHARING.md` before using this repository to support a manuscript claim.

## Analysis stages

- `Results/comparative/` and `Results/supporting/`: historical exploratory discovery, adapted comparator strategies, governance and repair; not fully nested independent confirmation.
- `Results/confirmatory/`: later locked-holdout reanalysis, with prior exposure to the wider research data/registry acknowledged.
- `Results/faithful_comparators/`: a separate protocol exercise. SliceLine uses its official package; SliceFinder and AFISP are compatibility/protocol reproductions, not all official turnkey implementations.
- `Results/governed_leakage_safe/`: internal grouped summaries only; RIPAS outputs are withheld.
- `Results/external_validation/`: source-cohort nested results/rankings only; RIPAS outputs are withheld.
- `Results/nhamcs_external_validation/`: stored NHAMCS aggregate evaluation. Source records are not redistributed; the script has source-training dependencies described in the run guide.

The corrected attribute-free comparison was newly run with shared folds and fresh training-only selection. Its joint robustness endpoint failed in all four families. Other stored numerical results were not recomputed or silently replaced during this update. This snapshot does not establish demographic fairness, universal repair benefit, patient-level independence, prospective clinical effectiveness or deployment readiness.

- `Results/attribute_free_corrected/`: verified new repair comparison; see `Documentation/ATTRIBUTE_FREE_CORRECTION.md`.
- `Results/attribute_free_historical/`: superseded summary and separate historical concealed-rule construct summary.

## Data access and verification

Obtain Kaggle data independently from the [original dataset record](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data). Redistribution rights remain unresolved; public accessibility does not itself establish permission to mirror raw or derived records. Obtain NHAMCS public-use records independently from [CDC/NCHS](https://www.cdc.gov/nchs/nhamcs/). RIPAS access cannot be granted by this repository.

Run `python tools/verify_release.py` from the root to check the snapshot. This is not an experimental rerun or institutional approval. Full reproduction needs authorized inputs and generated local intermediates; see `Documentation/RUN_GUIDE.md`.

## Reviewer access and citation

A private GitHub URL does not grant access. The owner must arrange access with named reviewers/editorial staff or provide a journal-approved confidential review package. Do not invite unknown recipients or make the repository public without a separate decision. A username-bearing link is not anonymous for double-blind review.

Use the repository commit corresponding to the submission, and identify withheld materials accurately. Do not state that all data or all manuscript results are publicly available. No open-source/data licence is assigned pending confirmation of ownership and institutional terms. Third-party components retain their own licences.

Research contact: Nazir Muhammad Ilyasu, School of Digital Science, Universiti Brunei Darussalam; 24h1711@ubd.edu.bn.
