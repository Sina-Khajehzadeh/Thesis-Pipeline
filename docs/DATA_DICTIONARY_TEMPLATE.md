# Dataset dictionary template

Complete one row for every input column before creating the final JSON
configuration. Add study-specific definitions where the supplied headings are
insufficient.

| Column | Study meaning | Pipeline role | Data type | Units or allowed values | Missing-value rule | Included as predictor? | Leakage or timing review | Notes |
|---|---|---|---|---|---|---|---|---|
| `outcome` | Define the event or state | target | binary | Define negative and positive labels | State exclusion or imputation rule | No | Confirm when outcome becomes observable | Map `positive_label` to class 1 |
| `subject_id` | Stable entity identifier | group | string/integer | Unique per subject or entity | Must not be missing when grouped splitting is required | No | Identifier only | Set `group_column`; otherwise remove this row |
| `observation_time` | Time of observation | timestamp | datetime | State timezone and precision | Define handling before splitting | Normally no | Confirm availability at prediction time | Set `timestamp_column`; otherwise remove this row |
| `record_id` | Row identifier | excluded identifier | string/integer | Unique per row | Must be unique if retained for audit | No | Identifier only | Add to `drop_columns` |
| `example_feature` | Replace with a substantive definition | numeric/categorical | replace | Replace | Replace | Yes/No | Confirm measured before prediction time | Replace |

## Required study decisions

- Unit of observation:
- Binary outcome definition:
- Positive class:
- Prediction time or index date:
- Intended prediction horizon:
- Independent rows, repeated entities, or temporal validation:
- Group column, if applicable:
- Timestamp column, if applicable:
- Permitted predictor list:
- Identifier, post-outcome, and leakage exclusions:
- Missing-data policy:
- Dataset version and provenance:
- Confidentiality classification:

The final decisions should agree with the JSON configuration, thesis methods,
and archived run artifacts.
