# Governed BASiC/ASAP research materials

Analysis code, output-free notebooks, slice-registry metadata, methodological documentation and screened non-RIPAS aggregate tables supporting governed hidden-failure slice discovery and targeted repair for emergency-department admission prediction.

BASiC means Bias-aware Auditable Slices in Clinical Pre-processing. ASAP means Auditable Slice Acceptance Protocol.

## Scope

This repository is maintained by `nazirkaoje777-dotcom`. The public-release documentation was prepared on 18 October 2026. It replaces a separate journal supplementary-file bundle with versioned shareable research materials. It is not a public patient dataset and does not contain every manuscript result: RIPAS-derived outputs are withheld. The manuscript Data Availability Statement identifies the exact supporting commit.

## Contents

- `Code/`: analysis scripts, including the fold-aligned attribute-free evaluation in `Code/attribute_free/`.
- `basic_slice_prem/Notebooks/`: ten analysis notebooks with outputs, attachments and execution metadata removed.
- `Registry/`: slice-level registry metadata and documented qualitative labels, not individual clinician questionnaire responses or adjudications.
- `Results/`: selected non-RIPAS aggregate tables grouped by analysis. Mixed and negative findings are retained.
- `Figures/attribute_free/`: vector figures for robustness and construct validation.
- `Documentation/`: execution requirements, parameters, output mapping, provenance and sharing boundaries.
- `RELEASE_MANIFEST.csv` and `tools/verify_release.py`: file sizes, checksums and an automated disclosure/integrity screen.

Start with `Documentation/RUN_GUIDE.md`. `Documentation/MANUSCRIPT_OUTPUT_MAP.csv` identifies public coverage and omissions for each manuscript analysis. Statistical implementation and evidence boundaries are described in `Documentation/REVIEW_STATUS.md`.

## Analysis provenance

The folders retain their established names so that existing scripts and provenance remain usable. `Results/attribute_free_corrected/` supplies the manuscript-supporting fixed-registry robustness evaluation; `Results/attribute_free_historical/` contains a labelled superseded summary and the separate concealed-clinical-rule construct summary. They are not interchangeable. The joint robustness endpoint was not met in any of the four model families.

Discovery/comparator and full-cohort repair outputs are exploratory; locked-holdout and grouped analyses reduce fitting/selection leakage but use a registry developed from the same source cohort. The comparator protocol exercise is not a claim that every method is an official turnkey implementation. No analysis establishes demographic fairness, universal repair benefit, patient-level independence, prospective clinical effectiveness or deployment readiness. Detailed provenance is retained for traceability, not presented as new experimental reproduction.

## Materials not released

No patient records, row-level outcomes or predictions, row indexes, profile identifiers, membership matrices, fold/selection assignments, fitted models, caches or runtime outputs are included. All RIPAS-derived result files and mixed manifests containing them remain excluded. Clinician responses, supervisor comments, correspondence, draft manuscripts, approval documents and third-party paper PDFs are also excluded.

Obtain Kaggle data independently from the [original dataset record](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data), and NHAMCS public-use records from [CDC/NCHS](https://www.cdc.gov/nchs/nhamcs/). Kaggle redistribution rights have not been established; public accessibility does not authorize mirroring source records. RIPAS access requires the responsible institution/data custodian's permissions and cannot be granted by this repository. See `Documentation/DATA_SHARING.md`.

## Verification and citation

Run `python tools/verify_release.py` from the repository root. This is a disclosure/integrity check, not institutional approval, an anonymity guarantee or an experimental rerun. Reproduction requires independently obtained authorized inputs and generated local intermediates.

Cite the exact commit named in the manuscript rather than assuming the moving `main` branch is the submitted version. The screened repository is intended for public access; a username-bearing repository is not an anonymous review archive.

## Reuse

Public visibility does not imply an open-source licence or permission to redistribute source data. No blanket licence is assigned to this mixed collection. Applicable author/institutional rights and third-party licences remain in force; contact the corresponding author where reuse requires permission.

Research contact: Nazir Muhammad Ilyasu, School of Digital Science, Universiti Brunei Darussalam; 24h1711@ubd.edu.bn.
