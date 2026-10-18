# Data access and reuse

This repository provides analysis code, clinical slice definitions, methodological documentation and aggregate tables. CSV tables can be opened in spreadsheet software. The [manuscript output map](MANUSCRIPT_OUTPUT_MAP.csv) identifies the files available for each analysis.

## Source data

- **Kaggle:** obtain the hospital triage and patient history data from the [original provider](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data). Source records and cleaned encounter-level derivatives are not redistributed. Users must check the provider's current terms before use.
- **NHAMCS:** obtain the public-use emergency-department files and documentation from [CDC/NCHS](https://www.cdc.gov/nchs/nhamcs/).
- **RIPAS:** data and derived result files are subject to institutional and data-custodian restrictions and are not distributed through this repository. Requests require the applicable permissions and approvals. Contact the corresponding author for the appropriate access route; access cannot be guaranteed.

Patient-level predictions, membership matrices and fold assignments are generated locally from authorized inputs and must remain outside the public repository. Research ethics approval and informed consent do not by themselves confer redistribution rights.

## Reuse

No repository-wide licence has been assigned. Applicable author and institutional rights and third-party terms remain in force. Contact Nazir Muhammad Ilyasu at nazirkaoje777@gmail.com or 24h1711@ubd.edu.bn about reuse that requires permission.

## Adding files

Before contributing, check that files contain no patient-level data, credentials or unauthorized derived outputs. `RELEASE_MANIFEST.csv` and `tools/verify_release.py` support integrity and basic disclosure checks; they do not replace an institutional data-release decision.
