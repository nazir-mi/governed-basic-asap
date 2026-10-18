# Evaluation metrics

The primary locked-test analysis uses seed 2026. The file paths below connect each reported measure to its analysis stage.

| Measure | Manuscript location | Supporting file |
| --- | --- | --- |
| Global AUROC, AUPRC, Brier score and recall | Section 4.5 and Table 9 | `Results/confirmatory/locked_holdout_global_results.csv` |
| Calibration intercept, slope and ECE10 | Sections 3.8 and 4.5; Table 9 | Primary-seed global rows and `primary_locked_test_calibration.csv` |
| Principal-slice AUPRC and Brier score | Section 4.5 and Table 9 | `Results/confirmatory/locked_holdout_slice_results.csv` |
| Profile-grouped internal performance | Section 4.6 | `Results/governed_leakage_safe/internal_pooled_oof_results.csv` and `internal_outer_fold_results.csv` |
| RIPAS transport and recalibration | Section 4.6; Figures 5–6 | Restricted external-validation outputs; see `DATA_SHARING.md` |
| NHAMCS external performance | Section 4.6; Table 11 | `Results/nhamcs_external_validation/nhamcs_external_results.csv` |
| Fixed-registry robustness | Table 12 and Figure 8 | `Results/attribute_free_corrected/` |
| Concealed-rule recovery and repair | Figure 9 | `Results/attribute_free_historical/concealed_group_experiment_summary.csv` |

## Locked-test calibration

`calibration_summary` in `Code/confirmatory_nested_holdout.py` regresses the binary outcome on the logit of clipped predicted probabilities. The fitted intercept and coefficient are the reported calibration intercept and slope. Ideal values are 0 and 1.

The implementation calls `calibration_curve(..., n_bins=10, strategy="quantile")` and reports the unweighted mean absolute difference between observed and predicted bin risks as ECE10. Tied probabilities can produce fewer than ten nonempty bins. Lower ECE10 is better.

The primary operating threshold is 0.315. The repaired-model calibration slope is 1.00614. The compact calibration CSV selects the corresponding seed-2026 values from the global-results table.

## Grouped and external analyses

The grouped transport workflow uses an encounter-count-weighted absolute error over quantile bins, whereas locked-test ECE10 uses an unweighted bin mean. These stage-specific summaries are not interchangeable.

Figure 6 compares mean transported risk with observed RIPAS prevalence. It is not a reliability curve or a complete calibration parameter table.

`Results/governed_leakage_safe/internal_fold_mean_effects.csv` contains BASiC/ASAP-minus-ERM differences between five outer-fold means. In LR, RF, XGBoost and LightGBM order, sensitivity differences are +0.0269, +0.0145, +0.0078 and +0.0058; AUPRC differences are −0.0045, −0.0054, −0.0036 and −0.0045; Brier differences are +0.0015, +0.0006, +0.0005 and +0.0008. These are internal fold-mean effects, not pooled-OOF or external-site effects.

## Direction of benefit

Higher AUROC, AUPRC and sensitivity indicate better performance on their respective measures. Lower Brier score, calibration error and CVaR log-loss are preferable. A sensitivity increase can coexist with poorer AUPRC or Brier score, so repair must be interpreted across all reported endpoints and the specified global tolerances.
