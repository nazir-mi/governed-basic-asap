# Inputs and reproduction boundaries

Most `Results/` folders retain stored historical outputs. `Results/attribute_free_corrected/` contains the new verified fold-aligned rerun and must be distinguished from the superseded historical comparison. Packages in `requirements.txt` are compatibility ranges (with selected packages pinned), not the exact original environment lockfile. No clean-environment end-to-end reproduction is claimed.

## Local authorized inputs â€” ignored by git

```
basic_slice_prem/
  Data/Raw/
    Kaggle_Triage_Modified.xlsx
    triage_cleaned_working.csv
    triage_train_diagnostics.csv
    approved_slice_registry_FINAL.csv
  Notebooks/
    slice_membership.csv
```

The modified workbook is a historical intermediate, not necessarily the original Kaggle file. Preserve preprocessing lineage and row order. Copy the frozen registry from `Registry/` to the private runtime folder where required. No encounter-level sample data is fabricated or supplied.

Run notebooks with working directory `basic_slice_prem/Notebooks/`, in numerical order including 05b for adapted-method summaries. They retain historical intermediate analyses and local dependencies. Rebuilding the frozen qualitative review judgments requires their historical annotations; numeric rules alone do not generate them.

Run standalone scripts from the repository root:

- `python Code/confirmatory_nested_holdout.py`: derived cohort, reconstructed membership and frozen registry.
- `python Code/faithful_comparators.py`: diagnostic training rows, plus `basic_slice_prem/Results/table_X_comparative_method_summary.csv` for reference-only adapted results. Generate that aggregate with the notebooks or copy the supplied same-named comparative aggregate into this local runtime path.
- `python Code/governed_leakage_safe_evaluation.py`: primary grouped workflow; requires local authorized RIPAS input set via `RIPAS_DATA_PATH`. The script generates sensitive row-level predictions locally. Never upload its runtime output folder. Use a fresh output directory for each full run because prediction files are appended.
- `python Code/multimodel_external_validation.py`: earlier/secondary nested and transport workflow, also requiring authorized `RIPAS_DATA_PATH`.
- `python Code/nhamcs_external_validation.py`: separately obtained CDC public-use file set via `NHAMCS_DATA_PATH`; consumes the preceding script's saved source rankings/thresholds. Its imported base module also requires `RIPAS_DATA_PATH`. Without authorized private inputs those dependencies cannot be reproduced from this repository alone.

The corrected attribute-free driver is supplied in `Code/attribute_free/`. It performs fresh training-only selection and does not import settings from the historical log. See `ATTRIBUTE_FREE_CORRECTION.md`.

Changes during packaging were limited to local path constants, notebook metadata/output removal and documentation. Statistical logic was not silently amended. Numerical tables were copied unchanged. Provenance and distributed-file hashes are included.


## Corrected attribute-free rerun

Use `Code/attribute_free/resolve_attribute_free.py`, not an imported historical selection log. The full command order, inputs, versions and interpretation limits are in `ATTRIBUTE_FREE_CORRECTION.md`. Local encounter-level outputs must remain in ignored `Local_Outputs/`.
