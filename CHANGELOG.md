# Changelog

## Reader-facing consistency update

- Documented the PhysioNet origin and local convenience name of
  `LMU_Final_Cleaned_Data.pkl`.
- Added a study-specific data dictionary and clarified future-glucose outcome,
  encoding, missing-value, group-isolation, and rerun-target semantics.
- Updated both notebooks to use files that remain in the repository.
- Added the study-specific tests to GitHub Actions.
- Corrected remaining V12/V13 labels in the loader and generic output path.

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
- Added a reusable generic base configuration and the machine-readable field
  dictionary.
- Added dependency lists, security guidance, and contribution guidance.
- Added lightweight repository checks and a GitHub Actions validation workflow.
- Retained explicit self-test, dry-run, run-validation, resume, and
  plot-regeneration entry points.

Earlier V12 and V13 files are development history and are not the authoritative
implementation.
