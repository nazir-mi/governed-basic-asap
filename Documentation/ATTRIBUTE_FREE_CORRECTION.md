# Attribute-free robustness protocol

`Results/attribute_free_corrected/` contains the fixed-registry robustness comparison used in Table 12 and Figure 8. Differences are BASiC/ASAP minus ERM between equal-weight means of five fold-level metrics. Figure 9 uses the separate concealed-clinical-rule construct summary.

## Evaluation settings

- Model families: logistic regression, Random Forest, XGBoost and LightGBM.
- Shared five-fold StratifiedGroupKFold assignment, grouping identical predictor profiles.
- Grouped 20% inner selection subset and three-fold grouped OOF ranking within the inner fit data.
- Repair grid: λ ∈ {0.5, 1, 2} and K ∈ {3, 5}, with ERM fallback if no positive-weight option is admissible.
- Selection objective: highest mean selected-slice sensitivity subject to AUPRC loss ≤0.005 and Brier increase ≤0.010. Ties are broken by higher AUPRC, then lower Brier.
- Repair influence: 0.60 normalized harm, 0.30 dual provenance and 0.10 normalized support.
- Ranking and operating threshold are computed from outer-training OOF predictions.
- Positive-case-only weighting, normalized harm floor 0.1 and weight cap 5; matched estimator seeds across arms.
- Joint endpoint: worst-registry-slice AUPRC delta >0, 10% CVaR delta <0, global AUPRC delta ≥−0.005 and global Brier delta ≤0.010.
- Uncertainty: 400 paired profile-cluster bootstrap replicates per outer test fold.

The clinical registry is fixed. This design evaluates repair on that registry, rather than independently confirming the entire discovery procedure. Focal-style and CVaR-style arms use two-stage sample-weighting approximations. Confidence intervals are conditional and pointwise, not selection-aware or simultaneous.

## Inputs and commands

Use Python 3.12 and `Code/attribute_free/requirements.txt`. Construct these row-aligned inputs from authorized data:

```text
basic_slice_prem/Data/Raw/triage_cleaned_working.csv
basic_slice_prem/Notebooks/slice_membership.csv
```

Reconstruct membership from the registry definitions and check its alignment with the source records. From the repository root, run:

```bash
python Code/attribute_free/resolve_attribute_free.py prepare
python Code/attribute_free/resolve_attribute_free.py run
python Code/attribute_free/resolve_attribute_free.py summarize
python Code/attribute_free/attribute_free_uncertainty.py
```

Outputs are written to `Local_Outputs/attribute_free_fold_aligned/`. This directory includes local fold assignments, selection roles and individual predictions and must not be published.

## Output interpretation

The aggregate results contain 20 model/fold evaluations, 100 arm/fold score records and 64 interval records. `protocol.json` records analysis settings and input checksums; `verification.json` summarizes alignment checks. These records describe the completed analysis. Running the commands above performs model fitting again and may require substantial computation.

The superseded model summary in `Results/attribute_free_historical/` is retained for version traceability and must not be substituted for the fold-aligned results. The concealed-rule experiment in that directory is a separate exploratory construct-validation analysis.
