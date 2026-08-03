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

The raw PhysioNet/MIMIC-derived data and generated model table are not included
in the repository. The preparation script does not require a pre-recorded
dataset hash. V14 records run-specific fingerprints in its generated audit
manifest without embedding a fixed expected fingerprint in this study code.

## Prepare the model table

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

## Common experiment definition

`configs/common_config.json` supplies the settings shared by every scenario:

- 20 Monte Carlo iterations and base seed 2025;
- sample sizes 1,000, 3,000, 10,000, 15,000, 25,000, 35,000, 40,000,
  46,000, 50,000, 60,000, 86,000, and the full prepared sample;
- original-prevalence patient-group sampling;
- strict stratified group splitting with 80% outer training and 20% test;
- a patient-disjoint 20% validation split within outer training;
- the same preprocessed 13-column numeric matrix; and
- a maximum of 50 Optuna trials for tuned conventional models.

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

## Automated checks

The tests do not fit the six models. They verify the preparation schema,
alignment, missing-value handling, patient-block integrity, shared settings,
device rules, budget references, and the fixed 0.29 context definition.

```bash
python -m unittest discover \
  -s thesis_study_blood_glucose/tests \
  -p "test_*.py"
```
