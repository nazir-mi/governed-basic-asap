# Governed BASiC/ASAP

BASiC/ASAP is an auditable framework for identifying clinically interpretable subgroups in which emergency-department admission models underperform and evaluating targeted model repair. It combines clinical rules with error-guided slice discovery, applies explicit eligibility checks, and prioritizes qualified slices for weighted retraining.

BASiC stands for **Bias-aware Auditable Slices in Clinical Pre-processing**. ASAP stands for **Auditable Slice Acceptance Protocol**.

## Research overview

The framework separates finding a poorly performing subgroup from deciding whether that subgroup is a credible intervention target. Clinical plausibility, stability, artifact checks and redundancy control inform eligibility. Repair influence combines normalized harm (0.60), dual-source provenance (0.30) and normalized support (0.10).

Evaluation covers logistic regression, Random Forest, XGBoost and LightGBM, with global performance, slice-level performance, tail loss and calibration reported separately. The study includes internal evaluation and external validation using RIPAS and NHAMCS cohorts. Targeted weighting is evaluated as a redistribution of predictive utility; improvement in every model or metric is not assumed.

## Repository contents

| Directory | Contents |
| --- | --- |
| `Code/` | Analysis, comparator, repair and validation scripts |
| `basic_slice_prem/Notebooks/` | Notebooks for preprocessing, discovery and governance analyses |
| `Registry/` | Clinical slice definitions and development annotations |
| `Results/` | Aggregate tables organized by analysis |
| `Figures/attribute_free/` | Vector figures for robustness and construct validation |
| `Documentation/` | Run instructions, parameters, metric definitions and manuscript output map |

Start with the [run guide](Documentation/RUN_GUIDE.md). The [manuscript output map](Documentation/MANUSCRIPT_OUTPUT_MAP.csv) links tables and figures to their supporting files. See the [evaluation design](Documentation/REVIEW_STATUS.md) for the role of each analysis and the [attribute-free protocol](Documentation/ATTRIBUTE_FREE_CORRECTION.md) for the fixed-registry robustness comparison.

## Data access

Obtain the Kaggle source data from the [original dataset record](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data) and NHAMCS public-use data from [CDC/NCHS](https://www.cdc.gov/nchs/nhamcs/). RIPAS data and derived files require institutional and data-custodian authorization and are not distributed here. Patient-level data are not included. See [data access and reuse](Documentation/DATA_SHARING.md).

## File verification

`RELEASE_MANIFEST.csv` records file sizes and SHA-256 checksums. To check the files, run:

```bash
python tools/verify_release.py
```

The check verifies file integrity and applies basic disclosure checks; it does not rerun the experiments. The analysis scripts require the inputs described in the run guide.

## Contact

Nazir Muhammad Ilyasu  
School of Digital Science, Universiti Brunei Darussalam  
Email: [nazirkaoje777@gmail.com](mailto:nazirkaoje777@gmail.com)  
Institutional email: [24h1711@ubd.edu.bn](mailto:24h1711@ubd.edu.bn)

Repository: https://github.com/nazir-mi/governed-basic-asap
