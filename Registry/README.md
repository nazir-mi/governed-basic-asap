# Clinical slice registry

`approved_slice_registry_FINAL.csv` contains the 29-row development registry used by the analysis workflow. Each row describes a candidate clinical slice. The file contains slice-level summaries and annotations, rather than patient membership records.

Columns such as `manual_plausibility`, `artifact_flag`, `redundancy_flag` and `final_accept` record the development-stage judgments and governance decisions used in the workflow. These annotations are not independent clinical-validation scores.

Candidate counts differ across analysis stages because eligibility checks, temperature-rule exclusions and redundancy control remove or combine definitions. Use the stage-specific tables and code when interpreting the number of candidates, qualified slices and repair targets.
