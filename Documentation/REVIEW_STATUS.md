# Scientific review and repository scope

## Attribute-free extension: quarantined, not corrected

The historical governed script assigns outer folds through deterministic profile-hash `grouped_splits`. The historical attribute-free extension instead uses `StratifiedGroupKFold` and retrieves repair strength, top-K and thresholds from the governed selection log by model name and fold number. The same fold number does not prove the same train/test records. Outcome-dependent settings may therefore have been selected using records that later appear in an extension test fold.

This is a concrete code-level concern, not yet a quantified estimate of its effect on results. The historical extension code and entire result bundle remain local and are excluded from this snapshot. Their exclusion is a quality-control quarantine, not selective removal of an unfavourable result. No corrected fairness result is claimed.

### Required repair before reinstatement

1. Save and verify a single outer-fold assignment indexed to the source row order and exact-profile groups; assert no profile crosses train/test.
2. For every model and outer fold, perform all repair selection and threshold selection exclusively within that outer training set. Do not import outcome-dependent settings from differently assigned folds.
3. Train ERM, BASiC/ASAP and all comparator arms using those same outer splits. Keep imputation, scaling, OOF weighting, ranking and any calibration inside the appropriate training partitions.
4. Freeze predictions before evaluating the identical held-out rows and fixed registry; compute paired endpoint differences and the chosen uncertainty estimates.
5. Regenerate affected tables, figures and manuscript text from the new run, retain the superseded local evidence, and document changes without optimizing for a desired win.

Excluding a repository file does not correct the corresponding manuscript claim. Before submission, either complete this rerun or remove/qualify the affected comparison in the manuscript. Manuscript files were not edited as part of this upload.

## Other interpretation limits

The original registry and notebooks are exploratory and retain historical development decisions. Later locked-holdout reanalysis does not erase prior exposure to the overall registry. Exact-profile grouping is not patient-level grouping without patient identifiers. Clinical conjunction recovery is not a demographic-fairness validation. Clinical definition review does not demonstrate model safety or clinical benefit.

## Scope map

| Analysis | Repository status |
| --- | --- |
| Historical discovery, adapted comparators and governance | Stored aggregate outputs included |
| Historical repair and sensitivity analyses | Stored aggregate outputs included, labelled exploratory |
| Locked-holdout reanalysis | Stored aggregate outputs included |
| Comparator protocol exercise | Included with implementation-fidelity qualifications |
| Governed internal grouped evaluation | Internal summaries included |
| RIPAS transport, recalibration and uncertainty | Results withheld pending repository-sharing permission |
| Earlier source-cohort nested results | Included; not interchangeable with primary grouped analysis |
| NHAMCS external evaluation | Stored aggregate outputs included, no source records |
| Attribute-free extension and related result bundle | Withheld pending fold-aligned rerun |

This upload does not constitute a new clean-environment reproduction. Syntax, checksums and disclosure patterns are checked separately from scientific validity.
