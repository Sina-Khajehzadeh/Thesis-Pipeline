# V14 reproducible thesis experiment pipeline

This repository contains a JSON-driven pipeline for reproducible binary
classification experiments. It supports ordinary row-level data, repeated
observations grouped by subject, and temporally ordered data.

The six core model families are L-SLR, Augmented SLR, Random Forest, XGBoost,
CatBoost, and TabPFN. Model availability depends on the packages and hardware
installed by the user.

## Scope

V14 can be adapted to another tabular dataset when:

- the outcome is binary;
- rows can be represented as tabular predictors;
- any subject/group identifier is supplied when observations are not
  independent; and
- temporal ordering is configured when the test partition must occur before or
  after the training partition.

Multiclass classification, regression, image data, text generation, and
survival analysis require methodological changes and are outside the current
implementation.

## Start here

1. Read [the pipeline reader guide](V14_Thesis_Pipeline_Reader_Guide.ipynb).
2. Use [the configuration dictionary](V14_Configuration_Dictionary.ipynb) to
   map a new dataset to JSON fields.
3. Complete [the dataset dictionary](docs/DATA_DICTIONARY_TEMPLATE.md) to record
   column roles, outcome coding, and leakage decisions.
4. Copy
   [the dataset template](configs/V14_dataset_template.example.json) and replace
   its placeholders.
5. Run a dry run before fitting any models.

## Installation

Create a clean Python environment and install the core dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
```

Optional models and output formats are listed in
[`requirements-optional.txt`](requirements-optional.txt). Install only the
packages required by the selected configuration.

## Dataset configuration

The pipeline accepts CSV, TSV, Parquet, Pickle, Feather, JSON, an in-memory
`pandas.DataFrame`, or a supported scikit-learn demonstration dataset. The
following fragments show the fields to change in a copy of the supplied
template; they are not complete standalone configurations.

Ordinary independent rows:

```json
{
  "data": {
    "source": "data/analysis.csv",
    "target": "outcome",
    "group_column": null,
    "drop_columns": ["record_id"]
  },
  "splitting": {
    "strategy": "stratified",
    "require_groups": false
  }
}
```

Repeated observations from the same subject:

```json
{
  "data": {
    "source": "data/analysis.parquet",
    "target": "outcome",
    "group_column": "patient_id"
  },
  "splitting": {
    "strategy": "stratified_group",
    "require_groups": true
  }
}
```

Temporally ordered observations:

```json
{
  "data": {
    "source": "data/analysis.parquet",
    "target": "outcome",
    "timestamp_column": "observation_time",
    "datetime_columns": ["observation_time"]
  },
  "splitting": {
    "strategy": "temporal",
    "temporal_window": "latest"
  }
}
```

Do not commit confidential data, credentials, fitted models, or experiment
outputs to the repository.

## Commands

Implementation checks:

```bash
python V14_Thesis_Pipeline.py --self-test
```

Configuration and dependency dry run:

```bash
python V14_Thesis_Pipeline.py --config path/to/experiment.json --dry-run
```

Run one configuration:

```bash
python V14_Thesis_Pipeline.py --config path/to/experiment.json
```

Run all enabled configurations in a directory:

```bash
python V14_Thesis_Pipeline.py --config-dir path/to/configuration_directory
```

Resume a run:

```bash
python V14_Thesis_Pipeline.py --config path/to/experiment.json --resume --run-dir path/to/run
```

Validate an existing run:

```bash
python V14_Thesis_Pipeline.py --validate-run --run-dir path/to/run
```

Regenerate figures without refitting models:

```bash
python V14_Thesis_Pipeline.py --regenerate-plots --run-dir path/to/run
```

## Repository layout

```text
V14_Thesis_Pipeline.py                  main command-line pipeline
dataset_loader.py                       dataset loading and binary target mapping
V14_Thesis_Pipeline_Reader_Guide.ipynb  methodological walkthrough
V14_Configuration_Dictionary.ipynb      field dictionary and adaptation workflow
docs/V14_CONFIGURATION_DICTIONARY.json  machine-readable field reference
docs/DATA_DICTIONARY_TEMPLATE.md         study-specific column definition form
configs/                                inheritance-based JSON examples
requirements-core.txt                   required CPU/runtime packages
requirements-optional.txt               model/output-specific packages
tests/                                  automated pipeline tests
.github/workflows/validate.yml           GitHub validation workflow
SECURITY.md                              data and credential guidance
CONTRIBUTING.md                          change and validation expectations
CHANGELOG.md                              public-version history
CITATION.cff.template                     citation metadata template
```

## Reproducibility

Every definitive run should retain:

- the exact JSON configuration;
- source and loader checksums;
- the dataset identifier and permitted predictor list;
- split indices and split audits;
- model parameters and Optuna trials;
- predictions and recomputed metrics;
- runtime, device, CPU, and energy-scope evidence;
- the artifact index; and
- a passing validation report.

The tuned-reference HPO budget and total model runtime are separate quantities.
See the reader guide before interpreting runtime comparisons.

## Security and credentials

The repository contains no embedded API key or token. Credentials for optional
external services must be supplied through the user's environment and must
never be committed. See [`SECURITY.md`](SECURITY.md).

## Citation and license

Before public release, complete the citation metadata and select a software
license. Rename and complete `CITATION.cff.template` as `CITATION.cff`. These
are owner decisions. Review the
[publication manifest](docs/PUBLICATION_MANIFEST.md) and
[release checklist](docs/RELEASE_CHECKLIST.md) before the first commit.
