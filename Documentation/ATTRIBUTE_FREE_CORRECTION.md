# Corrected attribute free evaluation

The default repair comparison is `Results/attribute_free_corrected/`.
Historical summaries are kept separately and are not substitutes for the new run.
Table 12 uses BASiC/ASAP-minus-ERM differences between equal-weight means of five
fold metrics. Full arm summaries and pointwise conditional confidence intervals
are provided; Figure 8 displays the corrected weighting comparisons. Figure 9
uses the separate historical concealed-clinical-group summary.

## Frozen analysis choices

- Four families: logistic regression, Random Forest, XGBoost and LightGBM.
- One saved five-fold StratifiedGroupKFold assignment shared by all arms/models.
- Inner grouped 20% selection subset; three-fold grouped OOF ranking in inner fit data.
- λ in {0.5, 1, 2}, K in {3, 5}, plus ERM fallback if no positive-weight option is admissible.
- Select greatest mean selected-slice sensitivity among options with AUPRC loss ≤0.005 and Brier increase ≤0.010; break ties by AUPRC then lower Brier.
- Ranking: 0.60 normalized harm, 0.30 dual provenance, 0.10 normalized support.
- Recompute ranking and decision threshold on outer-training OOF predictions.
- Positive-case-only weighting, normalized harm floor 0.1, cap 5; identical estimator seed across arms.
- Joint endpoint: worst-registry AUPRC delta >0, 10% CVaR delta <0, global AUPRC delta ≥−0.005, global Brier delta ≤0.010.
- 400 paired profile-cluster bootstrap samples within each outer test fold; pointwise conditional intervals, not selection-aware or simultaneous intervals.

The focal-style and CVaR-style arms are fixed two-stage sample-weighting
approximations, not custom differentiable focal or formal DRO objectives.
The clinical registry was previously developed and stays fixed; this correction
does not independently confirm the entire discovery procedure.

## Reproduction

Use Python 3.12 and the versions in `Code/attribute_free/requirements.txt`.
Obtain authorized source data independently and provide
`basic_slice_prem/Data/Raw/triage_cleaned_working.csv` and a reconstructed,
row-aligned `basic_slice_prem/Notebooks/slice_membership.csv`. Source records and
the membership matrix are intentionally not distributed. Use the supplied
registry/code and the existing membership-alignment checks to reconstruct them.

From the repository root run:

```
python Code/attribute_free/resolve_attribute_free.py prepare
python Code/attribute_free/resolve_attribute_free.py run
python Code/attribute_free/resolve_attribute_free.py summarize
python Code/attribute_free/attribute_free_uncertainty.py
```

New outputs, including local-only fold assignments, inner selection roles and
individual predictions, go to ignored `Local_Outputs/attribute_free_fold_aligned/`.
Never upload that whole directory. The portable driver does not require the
incomplete historical prediction export. Statistical fitting, selection and
scoring functions are unchanged from the executed corrected driver; only path
resolution and historical audit-input requirements were adapted. Numerical
helper definitions were extracted without altering their function bodies.

`protocol.json` records the original local run inputs and their checksums;
those paths are provenance, not downloadable records. `verification.json`
reports the local checks without disclosing row identifiers. The completed
rerun includes all 20 model/fold evaluations, 100 arm/fold score records and
64 interval records. Local execution notes are summarized below.

The run resumed from completed checkpoints after a process ended before
XGBoost output. A failed parallel worker produced no fitted-model checkpoint.
The resumed loader read only required source columns; reconstructed outcomes
and predictor-profile hashes matched the frozen manifest. Settings were not
changed in response to interim outcomes.
