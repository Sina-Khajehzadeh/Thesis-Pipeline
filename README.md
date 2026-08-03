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
2. For a new dataset, begin with
   [`configs/base_config.example.json`](configs/base_config.example.json).
3. For the submitted blood-glucose study, follow
   [the study-specific guide](thesis_study_blood_glucose/README.md).
4. Select one of the three study configurations documented in that guide.
5. Run a dry run before fitting any models.

## Submitted blood-glucose thesis study

The exact shared preparation and three scenario configurations used for the
blood-glucose management study are documented in
[the study-specific guide](thesis_study_blood_glucose/README.md). These files
set a standardized rerun target of 20 iterations, use the common 13-feature
model table, enforce strict patient-group isolation, and retain the fixed 0.29
TabPFN context used in the XGBoost-referenced scenario. The study
configurations inherit from the generic template
`configs/base_config.example.json`, but override its illustrative iteration,
data, device, budgeting, and output values. All authoritative study settings
are kept in the study folder.

The repository contains code and configurations, not the protected dataset or
the thesis result tables. A rerun can reproduce the design and audit trail;
runtime and energy values can differ with hardware, package versions, and
remote-service conditions.

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

To run all three blood-glucose scenarios, install both requirement files:

```powershell
pip install -r requirements-core.txt
pip install -r requirements-optional.txt
```

The requirement files describe the supported package set rather than an exact
historical lock. Definitive runs should retain the package versions written to
their environment manifest. The cloud/client TabPFN scenarios additionally
depend on service availability and the model version supplied by that service.

## Dataset configuration

The pipeline accepts CSV, TSV, Parquet, Pickle, Feather, JSON, an in-memory
`pandas.DataFrame`, or a supported scikit-learn demonstration dataset. The
following fragments illustrate the main fields; they are not complete
standalone configurations.

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
configs/base_config.example.json         generic template; not a thesis scenario
thesis_study_blood_glucose/              submitted study preparation and scenarios
thesis_study_blood_glucose/DATA_DICTIONARY.md  study variables and encodings
requirements-core.txt                   required CPU/runtime packages
requirements-optional.txt               model/output-specific packages
tests/                                  automated software validation checks
.github/workflows/validate.yml           GitHub validation workflow
SECURITY.md                              data and credential guidance
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

The three study JSON files request 20 Monte Carlo iterations. This is the
standardized rerun target and should not be interpreted as a claim that every
historical setting completed 20 iterations in the originally submitted runs.

## Security and credentials

The repository contains no embedded API key or token. Credentials for optional
external services must be supplied through the user's environment and must
never be committed. See [`SECURITY.md`](SECURITY.md).

## Citation

When citing the software, report the repository URL, pipeline version V14, and
the exact Git commit used for the analysis.
