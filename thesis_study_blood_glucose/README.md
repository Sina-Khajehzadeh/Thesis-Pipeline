# Thesis study: blood-glucose management in intensive care

This directory contains the study-specific preparation and configurations used
to reproduce the three computational scenarios described in the thesis. The
general V14 pipeline remains dataset-agnostic; these files define the exact
blood-glucose study inputs and scenario rules.

## Shared analytical data

All three scenarios use one prepared table with:

- 141,430 insulin-event observations in the submitted analysis;
- `y_glucose_normal` as the binary outcome, equal to 1 when the first glucose
  measurement strictly after an insulin event is between 70 and 180 mg/dL;
- `SUBJECT_ID` as the patient-group variable; and
- the same 13 encoded numeric predictors, in a fixed order.

The source PhysioNet data and generated model table are not included in the
repository. The preparation script does not require a pre-recorded dataset
hash. V14 records run-specific fingerprints in its generated audit manifest
without embedding a fixed expected fingerprint in this study code.

## Data source and citation

The study is based on *Curated Data for Describing Blood Glucose Management in
the Intensive Care Unit* (version 1.0.1), available through PhysioNet and
derived from MIMIC-III version 1.4. Access must be obtained directly from
PhysioNet and remains subject to its credentialing and data-use requirements.
This repository does not redistribute the source data.

`LMU_Final_Cleaned_Data.pkl` is the local filename used in the thesis
workspace for the study extract obtained from the PhysioNet resource. The
credentialed PhysioNet table was saved in pandas Pickle format and renamed to
this filename for convenient loading in the preparation and scenario scripts.
The filename does not identify a separate database and does not change the
dataset's PhysioNet provenance. Readers must obtain the source resource under
their own PhysioNet authorization and create an equivalent local file.

Please cite the dataset and its accompanying data descriptor:

- Robles Arévalo A, Mateo-Collado R, Celi LA. *Curated Data for Describing
  Blood Glucose Management in the Intensive Care Unit* (version 1.0.1)
  [dataset]. PhysioNet; 2021.
  [https://doi.org/10.13026/517s-2q57](https://doi.org/10.13026/517s-2q57)
- Robles Arévalo A, Maley JH, Baker L, da Silva Vieira SM, da Costa Sousa JM,
  Finkelstein S, et al. Data-driven curation process for describing the blood
  glucose management in the intensive care unit. *Scientific Data*.
  2021;8:80.
  [https://doi.org/10.1038/s41597-021-00864-4](https://doi.org/10.1038/s41597-021-00864-4)

## Prepare the model table

The input Pickle must contain the columns documented in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). At minimum, these include the
patient identifier, insulin-event time and variables, glucose time and value,
and the PhysioNet paired-glucose fields. The submitted input contained 603,761
rows; the preparation produced 141,430 eligible insulin events from 9,264
patients. These counts are study audit values, not hard-coded acceptance rules.

From the repository root, run:

```bash
python thesis_study_blood_glucose/prepare_blood_glucose_data.py \
  --input "/path/to/LMU_Final_Cleaned_Data.pkl"
```

The default output is:

```text
thesis_study_blood_glucose/data/prepared_blood_glucose_model_table.pkl
```

The preparation preserves complete patient blocks and chronological order
within each patient. For exact alignment with the submitted Colab preparation,
the saved table is not randomly reordered first. Seeded patient-group sampling
and train/test isolation are performed by V14 through the common configuration.
The optional `--shuffle-groups` flag exists for diagnostic use but is not part
of the thesis rerun command.

The PhysioNet fields `GLC_AL`, `GLCTIMER_AL`, and `GLCSOURCE_AL` describe the
glucose reading originally paired with an insulin event in the source
resource. This study constructs a different prediction outcome from the first
glucose measurement strictly after the insulin event. Future-glucose values and
timestamps are excluded from the predictor matrix.

## Common experiment definition

`configs/common_config.json` supplies the settings shared by every scenario:

- a standardized target of 20 Monte Carlo iterations and base seed 2025;
- sample sizes 1,000, 3,000, 10,000, 15,000, 25,000, 35,000, 40,000,
  46,000, 50,000, 60,000, 86,000, and the full prepared sample;
- original-prevalence patient-group sampling;
- strict stratified group splitting with 80% outer training and 20% test;
- a patient-disjoint 20% validation split within outer training;
- the same preprocessed 13-column numeric matrix; and
- a maximum of 50 Optuna trials for tuned conventional models.

Missing values in the seven numeric source predictors are represented as 0.0
in the prepared matrix, matching the submitted scenario preparation. The six
encoded indicator columns and their reference categories are defined in the
study data dictionary. These zeros are modelling representations and should
not automatically be interpreted as observed clinical measurements.

## Scenario configurations

| File | Thesis scenario | Defining rule |
|---|---|---|
| `scenario_1_no_budget.json` | No-time-budgeting control | No shared runtime limit; cloud/client TabPFN and CPU conventional models |
| `scenario_2_tabpfn_budget.json` | TabPFN-runtime budget | Cloud/client TabPFN execution time limits conventional-model HPO |
| `scenario_3_xgboost_fixed_029.json` | XGBoost-referenced budget | XGBoost HPO time is the reference; all models run on CPU and TabPFN uses `ceil(0.29 × actual sampled records)` as fixed context |

Scenario 3 does not use an adaptive 1.65 or 1.85 runtime multiplier. The fixed
0.29 context is always attempted, while the original XGBoost reference budget
and any effective TabPFN budget are retained separately in the run evidence.

## Validate and run

Check each configuration before a long run:

```bash
python V14_Thesis_Pipeline.py \
  --config thesis_study_blood_glucose/configs/scenario_1_no_budget.json \
  --dry-run
```

Replace the filename with either of the other scenario files as needed. Run all
three enabled configurations from the directory with:

```bash
python V14_Thesis_Pipeline.py \
  --config-dir thesis_study_blood_glucose/configs
```

Cloud/client TabPFN authentication must be supplied through the supported local
environment or client login mechanism. Credentials are not stored in these
files.

The repository reproduces the experimental definitions and evidence-writing
workflow; it does not guarantee identical wall-clock time, energy estimates, or
cloud outputs on future hardware and service versions. The saved environment
and run manifests are the authoritative record for a completed rerun.

## Automated checks

The tests do not fit the six models. They verify the preparation schema,
alignment, missing-value handling, patient-block integrity, shared settings,
device rules, budget references, and the fixed 0.29 context definition.

```bash
python -m unittest discover \
  -s thesis_study_blood_glucose/tests \
  -p "test_*.py"
```
