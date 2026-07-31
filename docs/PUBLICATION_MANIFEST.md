# Public repository manifest

## Include

- `V14_Thesis_Pipeline.py`
- `dataset_loader.py`
- `README.md`
- `V14_Thesis_Pipeline_Reader_Guide.ipynb`
- `V14_Configuration_Dictionary.ipynb`
- `configs/`
- `docs/`
- `tests/test_repository_assets.py`
- `.github/workflows/validate.yml`
- `.gitignore`
- `requirements-core.txt`
- `requirements-optional.txt`
- `SECURITY.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- the completed `CITATION.cff`
- the selected `LICENSE`

## Retain locally but do not publish

- V12 and V13 source files;
- `former pipelines/`;
- the V12 `experiment_config.py` and `run_experiment.py` wrapper;
- V12/V13 validation reports and reader documents;

## Review before including

- study-specific figures that have been cleared for publication;
- a de-identified example dataset, if the study permits one.

Development history is not required to reproduce V14.

## Exclude

- raw or derived confidential data;
- predictions, group identifiers, and split-index files unless disclosure has
  been reviewed;
- local reruns, dashboard source data, caches, and experiment outputs;
- credentials, private URLs, environment files, and fitted models.

Before the first commit, inspect `git status --short` and use
`git check-ignore -v <path>` for any file whose inclusion is uncertain.
