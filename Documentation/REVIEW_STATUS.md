# Evaluation design

The analyses evaluate distinct questions: which slices are discovered, which pass governance checks, and whether targeted weighting improves held-out predictive performance.

## Analysis map

| Analysis | Purpose and supporting files |
| --- | --- |
| Discovery and governance | Candidate discovery, qualification and prioritization; `Results/comparative/` |
| Full-cohort repair and sensitivity | Exploratory repair and parameter analyses; `Results/supporting/` |
| Locked holdout | Training-based selection and held-out evaluation; `Results/confirmatory/` |
| Comparator protocol | Method-specific discovery procedures and common qualification; `Results/faithful_comparators/` |
| Grouped multi-model evaluation | Internal performance with predictor-profile grouping; `Results/governed_leakage_safe/` |
| External validation | RIPAS transport and NHAMCS evaluation; public NHAMCS aggregates in `Results/nhamcs_external_validation/`; RIPAS access is described in `DATA_SHARING.md` |
| Attribute-free robustness | Fixed-registry worst-slice AUPRC and tail log-loss; `Results/attribute_free_corrected/` |
| Concealed-rule construct validation | Recovery and repair of concealed clinical conjunctions; summary in `Results/attribute_free_historical/` |

Check [MANUSCRIPT_OUTPUT_MAP.csv](MANUSCRIPT_OUTPUT_MAP.csv) for exact paths and manuscript locations. Discovery and full-cohort repair are exploratory. Registry definitions were developed from the source cohort; grouped evaluation does not independently validate discovery.

## Attribute-free comparison

All four model families and five weighting arms share the same five-fold profile-grouped assignment. Repair strength and top-K are selected within each outer training set, and estimator seeds are matched across arms. Exact predictor profiles do not cross outer train/test or inner fit/selection boundaries.

The joint robustness endpoint was met in 0 of 4 model families. BASiC/ASAP-minus-ERM worst-registry-slice AUPRC differences were negative and 10% CVaR log-loss differences were positive in all four. Random Forest also exceeded the global AUPRC-loss tolerance.

`Results/attribute_free_historical/attribute_free_model_summary.csv` is a superseded comparison and is not the source for Table 12 or Figure 8. It used repair settings selected under different fold populations. The manuscript comparison instead performs aligned, training-only selection. The separate concealed-rule summary supplies Figure 9 and is not part of the robustness endpoint.

## Comparator interpretation

Adapted discovery strategies and the method-specific comparator protocol are separate analyses. Method labels, search spaces and implementation qualifications must be read together; a shared qualification procedure does not make every comparator an official turnkey implementation. Focal-style and CVaR-style weighting are two-stage approximations rather than custom differentiable objectives.

## Statistical interpretation

Attribute-free intervals use 400 paired predictor-profile cluster bootstrap replicates within each outer test fold and equal-weight five-fold metric means. They are pointwise, unadjusted for multiple comparisons and conditional on fitted predictions and fixed slice definitions. Model fitting and repair selection are not repeated in the bootstrap.

Predictor-profile grouping reduces one form of dependence but cannot establish patient-level independence without patient identifiers. Clinical-conjunction recovery evaluates discovery of specified constructs, not demographic fairness. The retrospective analyses do not establish universal repair benefit, deployment readiness or clinical effectiveness.
