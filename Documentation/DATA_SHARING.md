# Conservative repository-sharing policy

This repository is private but remains a third-party cloud transfer. Private visibility is not a substitute for institutional permission.

## Included material

Only methodological code, output-free notebooks, historical slice-level annotations and screened non-RIPAS aggregate outputs are supplied. These are provided for private scholarly review, not relicensed as public-domain or open-source material. Author/institutional ownership and reuse terms must be confirmed before any broader public release.

## Withheld material

- Every RIPAS record, prediction, membership, outcome, profile hash, detailed subgroup output and overall result table; no approval letter, data key or signature is uploaded.
- Mixed analysis manifests containing RIPAS-derived summaries.
- Raw and cleaned Kaggle records, intermediate diagnostic rows, row-level examples, OOF prediction tables and membership matrices; source redistribution rights remain unconfirmed.
- NHAMCS raw records; obtain them from the official source.
- Fitted models, caches and serialized objects, which have not been assessed for disclosure risk.
- Clinician respondents' identifiers, ratings, free text, adjudication records and private review communications.
- Supervisor feedback, correspondence, draft manuscripts and third-party article PDFs.

`WITHHELD_FILES.csv` identifies omissions from the previous candidate; the policy above also excludes material never admitted to that candidate. Non-inclusion of data must not be described as data being publicly downloadable from this repository.

## Future release checklist

Obtain explicit confirmation of which RIPAS aggregate outputs may be placed on GitHub, whether privately or publicly; MHREC approval alone is not used as that authorization. Confirm the research-use and redistribution terms for Kaggle separately. Confirm author/institutional rights and an appropriate licence for original code and tables. Assess overlapping groups, derived rates and complementary counts before adding future outputs.

The automated screen flags explicit group/outcome counts between 1 and 9 where recognized, plus complementary counts where both n and events are supplied. This is a conservative project screen, not an institutional suppression rule or an anonymity guarantee.

Do not add runtime output directories to git. A future accidentally committed restricted file must be treated as a disclosure incident even if subsequently deleted, because git history can retain it.
