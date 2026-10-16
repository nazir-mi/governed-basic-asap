# Evaluation metric reporting map

The primary Kaggle locked-test run uses seed 2026. Its stored outputs are in `Results/confirmatory/locked_holdout_global_results.csv`; `primary_locked_test_calibration.csv` is a compact transcription of those rows, not a rerun.

## Reported evaluation measures

| Measure | Manuscript location | Stored evidence |
| --- | --- | --- |
| Global AUROC, AUPRC, Brier score and recall | Section 4.5 and Table 9 | `Results/confirmatory/locked_holdout_global_results.csv` |
| Calibration intercept and slope | Section 3.8 definitions; Section 4.5 and Table 9 values | The same primary-seed global rows and compact calibration CSV |
| ECE10 | Section 3.8 implementation; Table 9 | `ECE10` in the same rows |
| Principal-slice AUPRC and Brier score | Section 4.5 and Table 9 | `Results/confirmatory/locked_holdout_slice_results.csv` |
| Profile-grouped internal summaries | Section 4.6; stage-specific analysis | `Results/governed_leakage_safe/internal_pooled_oof_results.csv` and `internal_outer_fold_results.csv` |
| RIPAS transported/recalibrated model summaries | Section 4.6 and Figures 5–6 | Controlled local outputs; not supplied in this repository pending permission |
| NHAMCS model summaries | Section 4.6 and Table 11 | `Results/nhamcs_external_validation/nhamcs_external_results.csv` |

`calibration_summary` in `Code/confirmatory_nested_holdout.py` fits a logistic regression of the binary outcome on the logit of clipped predicted probabilities; its fitted intercept and coefficient are the reported calibration intercept and slope. The code uses `calibration_curve(..., n_bins=10, strategy="quantile")` and the unweighted mean absolute difference between the returned observed and mean predicted bin risks for `ECE10`. Tied probabilities can produce fewer than ten nonempty bins. This implementation is not a claim of a universal binning convention. Ideal intercept and slope values are 0 and 1, and smaller ECE10 is better.

Figure 6 shows mean transported risk versus observed RIPAS prevalence, not a reliability curve and not the complete intercept/slope/error parameter table. Full RIPAS-derived tables remain withheld under `DATA_SHARING.md`.

## Reporting corrections dated 16 October 2026

The primary locked-test threshold is 0.315, not the 0.48 threshold from a different historical operating-point analysis. The repaired calibration slope is 1.00614, not 1.00006. The original stored results were not changed. The manuscript now includes intercept and ECE10 rows so that the evaluation methods and displayed results can be traced directly.

The earlier editorial corrections were followed by a new fold-aligned attribute-free rerun. Its corrected aggregates are in `Results/attribute_free_corrected/`; superseded values are separated from the default comparison. See `ATTRIBUTE_FREE_CORRECTION.md` and `REVIEW_STATUS.md`.

## Internal versus external effect attribution

`Results/governed_leakage_safe/internal_fold_mean_effects.csv` contains BASiC/ASAP minus ERM arithmetic differences between the five stored outer-fold means. In LR, RF, XGBoost and LightGBM order, sensitivity changes are +0.0269, +0.0145, +0.0078 and +0.0058; AUPRC changes are −0.0045, −0.0054, −0.0036 and −0.0045; and Brier changes are +0.0015, +0.0006, +0.0005 and +0.0008. These are neither pooled-OOF nor external-site effects.

The previous manuscript range 0.0065–0.0581 was attributed to internal sensitivity but came from the adult external-site comparison. The corrected internal range is 0.0058–0.0269. This is a cohort-attribution correction from existing saved metrics, not a changed outcome or experiment rerun. External-site tables remain withheld under the repository-sharing policy.

The primary grouped transport implementation differs from the primary locked-test ECE10 convention: it uses an encounter-count-weighted absolute error over quantile bins. The revised Methods distinguishes these stage-specific calibration summaries rather than claiming a single interchangeable ECE implementation.

Table 12 and Figure 8 now use the corrected attribute-free rerun. Figure 9 reports the separate historical concealed-rule construct experiment. Other stage-specific table numbers and numerical outputs are retained.
