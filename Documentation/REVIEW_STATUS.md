# Scientific review and repository scope

## Corrected attribute free repair evaluation

A new fold-aligned evaluation was completed on 17 October 2026. All four model
families and five weighting arms share a frozen, row-indexed outer-fold
assignment. Repair strength and top-K are selected afresh inside each outer
training set; no settings are imported from the historical selection log.
Estimator seeds are matched across arms. Exact predictor profiles do not cross
outer train/test or inner fit/selection boundaries. The local integrity audit
passed; its aggregate summary is included without encounter identifiers.

BASiC/ASAP met the retained joint robustness endpoint in 0 of 4 model families.
Worst-registry-slice AUPRC point differences were negative and 10% CVaR
log-loss differences positive in all four. Random Forest also exceeded the
global AUPRC-loss tolerance. These are robustness results, not demographic
fairness, deployment-readiness or clinical-effectiveness evidence.

## Historical provenance

The old attribute-free comparison reused outcome-dependent repair settings
by model and fold number despite different saved fold populations. A saved
governed prediction export is also incomplete and does not match the summary
fold sizes. Current scripts alone do not establish which historical splitter
produced each artifact; a specific splitter attribution is therefore withdrawn.
The corrected run does not depend on those predictions or settings.

Superseded aggregate summaries are retained in `Results/attribute_free_historical/`.
The separate concealed-clinical-group experiment remains historical and
exploratory; its 8/12 recovery and 3/12 repair-success counts were not rerun.
The corrected comparison changes fold alignment, refreshes selection and
matches estimator seeds, so differences are not an isolated causal estimate
of leakage removal alone. It is a retrospective re-evaluation of a previously
developed registry, not untouched confirmation of discovery.

## Uncertainty and interpretation

Corrected intervals use 400 paired exact-profile cluster bootstrap replicates
within outer test folds and equal-weight five-fold metric means. They are
pointwise, unadjusted for multiple comparisons and conditional on fitted
predictions and fixed definitions; training and selection are not repeated.
No patient identifier was available, so grouping cannot prove patient-level
independence. The fixed registry does not cover every possible disadvantaged
group, and clinical conjunction recovery is not demographic-fairness validation.

## Scope map

| Analysis | Repository status |
| --- | --- |
| Historical discovery, adapted comparators and governance | Aggregate outputs retained |
| Historical repair and sensitivity analyses | Exploratory outputs retained |
| Locked-holdout reanalysis | Stored aggregate outputs retained |
| Comparator protocol exercise | Implementation-fidelity qualifications retained |
| Governed internal grouped evaluation | Earlier internal summaries retained, not silently replaced |
| RIPAS transport, recalibration and uncertainty | Results withheld pending sharing permission |
| NHAMCS external evaluation | Stored aggregates, no source records |
| Corrected attribute-free repair comparison | New code, aggregate results and integrity audit included |
| Historical concealed-rule construct experiment | Summary retained, explicitly not rerun |

No patient records, individual predictions, group hashes, row assignments,
selection-role assignments or membership matrices are released. Only the
corrected attribute-free repair comparison was newly run; other analyses are
stored historical results, not a newly reproduced complete manuscript.
