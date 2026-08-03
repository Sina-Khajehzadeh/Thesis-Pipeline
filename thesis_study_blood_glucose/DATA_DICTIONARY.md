# Blood-glucose study data dictionary

This document defines the input contract and the final model columns used by
the three thesis scenarios. The source is *Curated Data for Describing Blood
Glucose Management in the Intensive Care Unit*, version 1.0.1, obtained from
PhysioNet under credentialed access.

## Local input file

`LMU_Final_Cleaned_Data.pkl` is a pandas Pickle representation of the local
study extract derived from the PhysioNet resource. It was renamed for
convenient loading in the thesis workspace. It is not a new database and is
not distributed in this repository.

The preparation script requires these source columns:

| Column | Role |
|---|---|
| `SUBJECT_ID` | De-identified patient identifier and group-isolation key |
| `STARTTIME` | Start time of the insulin event or changed infusion rate |
| `GLCTIMER` | Time of an individual glucose measurement |
| `GLC` | Individual glucose value in mg/dL; used to locate the future outcome |
| `GLCSOURCE` | Measurement method for `GLC` |
| `GLCTIMER_AL` | Time of the glucose reading paired by the PhysioNet source rules |
| `GLC_AL` | Source-paired glucose value in mg/dL |
| `GLCSOURCE_AL` | Measurement method for `GLC_AL` |
| `LOS_ICU_days` | ICU length of stay in days |
| `first_ICU_stay` | Indicator that the ICU stay is the first for the hospital admission |
| `INPUT` | Insulin bolus dose in units |
| `INPUT_HRS` | Insulin infusion rate in units per hour |
| `INFXSTOP` | Indicator that an insulin infusion was discontinued |
| `INSULINTYPE` | Source insulin acting-type category |
| `EVENT` | Source administration-route/event category |
| `RULE` | PhysioNet rule used to pair the source glucose and insulin event |

The script may accept additional columns but does not use them as predictors
unless they are listed below.

## Outcome and grouping

| Output | Definition | Model input? |
|---|---|---|
| `y_glucose_normal` | 1 when the first glucose strictly after an insulin event is 70-180 mg/dL inclusive; otherwise 0 | Outcome only |
| `SUBJECT_ID` | Patient group used for sampling and patient-disjoint splitting | No |
| `GLC_next` | Future glucose used to construct the outcome | No |
| `GLCTIMER_next` | Time of the future glucose | No |
| `GLCSOURCE_next` | Measurement source of the future glucose | No |

The future-glucose fields are constructed temporarily and excluded from the
saved feature matrix to prevent outcome leakage.

## Final 13 predictors

| Predictor | Definition and encoding |
|---|---|
| `LOS_ICU_days` | ICU length of stay in days; numeric |
| `first_ICU_stay` | First-ICU-stay indicator; numeric |
| `INPUT` | Insulin bolus dose in units; numeric |
| `INPUT_HRS` | Insulin infusion rate in units per hour; numeric |
| `INFXSTOP` | Infusion-discontinuation indicator; numeric |
| `GLC_AL` | Glucose value paired with the insulin event by the PhysioNet source rules, mg/dL; numeric |
| `RULE` | Numeric source pairing-rule identifier |
| `INSULINTYPE_Long` | 1 for long-acting insulin, otherwise 0 |
| `INSULINTYPE_Short` | 1 for short-acting insulin, otherwise 0 |
| `EVENT_BOLUS_PUSH` | 1 for an intravenous bolus event, otherwise 0 |
| `EVENT_INFUSION` | 1 for an infusion event, otherwise 0 |
| `GLCSOURCE_AL_FINGERSTICK` | 1 when the paired glucose was measured by fingerstick, otherwise 0 |
| `GLCSOURCE_AL_nan` | 1 when the paired-glucose measurement source is missing, otherwise 0 |

For the indicator pairs, all-zero values represent the reference or another
source category: intermediate/missing insulin type, non-push/non-infusion
event, or blood/other glucose source as applicable.

Missing values in the seven numeric predictors are converted to 0.0 before
model fitting, matching the submitted scenario preparation. This is a modelling
representation rather than evidence that a clinical zero was observed. All 13
predictors are stored as `float32`; the outcome is binary and the patient group
is retained separately.

## Submitted-study audit values

The submitted preparation produced:

- 603,761 rows in the local PhysioNet-derived input;
- 141,430 insulin events with a later glucose outcome;
- 9,264 unique patient groups;
- 13 predictors; and
- class-1 prevalence of approximately 0.694464.

These values help audit the submitted study. They are not hard-coded dataset
hashes and do not authorize redistribution of PhysioNet data.
