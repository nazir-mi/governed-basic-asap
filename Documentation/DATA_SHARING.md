# Repository sharing scope

The public repository contains screened research code and aggregate outputs, not a public patient dataset. Public visibility was requested by the corresponding author. That request does not expand rights over third-party or institutionally controlled data, which remain excluded.

## Included materials

Analysis scripts, output-free notebooks, slice-level registry annotations, methodological documentation, selected non-RIPAS aggregate result tables and vector figures are included. Tables are supplied in CSV format for direct reading in spreadsheet software. Source spreadsheets and serialized runtime files are not needed in the public package.

The manuscript output map identifies public coverage and omissions. Provenance distinguishes the manuscript-supporting attribute-free evaluation from superseded summaries and the separate construct-validation experiment. Negative and mixed findings are retained.

## Excluded materials

- RIPAS records, outcomes, predictions, profile hashes, membership matrices, subgroup outputs and overall result tables; mixed manifests containing RIPAS summaries; data keys, approval letters and signatures.
- Raw or cleaned Kaggle records, encounter-level diagnostic rows, examples, OOF prediction tables and membership matrices. Kaggle redistribution permission has not been established.
- NHAMCS source records, which must be obtained independently from CDC/NCHS.
- Row indexes, fold assignments, inner selection-role assignments, individual predictions, trained models, caches, serialized objects and generated local outputs.
- Clinician identifiers, individual ratings, free text and adjudication records; supervisor comments, correspondence, manuscript drafts and third-party article PDFs.

RIPAS research ethics approval and informed consent are not treated as permission to publish data or derived result files on GitHub. Requests for restricted materials must be referred to the corresponding author and the responsible institution/data custodian; availability is not promised without the required permissions.

## Integrity and future additions

`RELEASE_MANIFEST.csv` lists files, byte sizes and SHA-256 hashes. Run `python tools/verify_release.py` to check the manifest and project disclosure rules. The screen checks recognized record-level columns, notebook outputs, credential patterns, personal absolute paths and small explicit counts. It is not a substitute for institutional disclosure review or a guarantee against re-identification.

The pre-release Git history was checked as well as the current snapshot. Future additions must be screened before committing: deleting a restricted file from the current branch does not remove it from history. Never commit runtime output directories or RIPAS-derived files without explicit redistribution permission.

No blanket open-source or data licence is assigned to the collection. Public access is distinct from reuse permission; third-party licences and author/institutional rights remain applicable.
