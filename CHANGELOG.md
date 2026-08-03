# Changelog

## Blood-glucose thesis study package

- Added one common preparation script for the 13-feature blood-glucose model
  table without embedded expected dataset hashes.
- Added a shared 20-iteration configuration and separate no-budget,
  TabPFN-budget, and XGBoost fixed-0.29-context configurations.
- Added automated checks for preparation, group integrity, common settings,
  execution devices, budget references, and scenario-specific rules.

## V14 repository release

- Established `V14_Thesis_Pipeline.py` as the public thesis pipeline.
- Added a reader guide and a configuration-dictionary notebook.
- Added reusable inheritance-based JSON examples for independent, grouped, and
  temporal study designs.
- Added machine-readable field definitions, dependency lists, security
  guidance, contribution guidance, and a public-release checklist.
- Added lightweight repository checks and a GitHub Actions validation workflow.
- Retained explicit self-test, dry-run, run-validation, resume, and
  plot-regeneration entry points.

Earlier V12 and V13 files are development history and are not the authoritative
implementation.
