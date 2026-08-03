"""Prepare the common blood-glucose modelling table used by all scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "y_glucose_normal"
GROUP_COLUMN = "SUBJECT_ID"

EXPECTED_FEATURE_COLUMNS = [
    "LOS_ICU_days",
    "first_ICU_stay",
    "INPUT",
    "INPUT_HRS",
    "INFXSTOP",
    "GLC_AL",
    "RULE",
    "INSULINTYPE_Long",
    "INSULINTYPE_Short",
    "EVENT_BOLUS_PUSH",
    "EVENT_INFUSION",
    "GLCSOURCE_AL_FINGERSTICK",
    "GLCSOURCE_AL_nan",
]

NUMERIC_PREDICTORS = [
    "LOS_ICU_days",
    "first_ICU_stay",
    "INPUT",
    "INPUT_HRS",
    "INFXSTOP",
    "GLC_AL",
    "RULE",
]

CATEGORICAL_PREDICTORS = [
    "INSULINTYPE",
    "EVENT",
    "GLCSOURCE_AL",
]

REQUIRED_SOURCE_COLUMNS = sorted(
    {
        GROUP_COLUMN,
        "STARTTIME",
        "GLCTIMER",
        "GLCTIMER_AL",
        "GLC",
        "GLCSOURCE",
        *NUMERIC_PREDICTORS,
        *CATEGORICAL_PREDICTORS,
    }
)


def _category_indicator(series: pd.Series, value: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.eq(value).fillna(False).astype("float32")


def _attach_next_glucose(
    patient_insulin: pd.DataFrame,
    patient_glucose: pd.DataFrame,
) -> pd.DataFrame:
    insulin_times = patient_insulin["STARTTIME"].to_numpy()
    glucose_times = patient_glucose["GLCTIMER"].to_numpy()
    glucose_values = patient_glucose["GLC"].to_numpy(dtype=float)
    glucose_sources = patient_glucose["GLCSOURCE"].to_numpy()

    next_positions = np.searchsorted(
        glucose_times,
        insulin_times,
        side="right",
    )
    next_glucose = np.full(len(insulin_times), np.nan, dtype=float)
    next_time = np.full(
        len(insulin_times),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    next_source = np.full(len(insulin_times), None, dtype=object)
    valid = next_positions < len(glucose_times)
    next_glucose[valid] = glucose_values[next_positions[valid]]
    next_time[valid] = glucose_times[next_positions[valid]]
    next_source[valid] = glucose_sources[next_positions[valid]]

    result = patient_insulin.copy()
    result["GLC_next"] = next_glucose
    result["GLCTIMER_next"] = next_time
    result["GLCSOURCE_next"] = next_source
    return result


def _shuffle_complete_groups(
    frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    group_ids = frame[GROUP_COLUMN].drop_duplicates().to_numpy()
    shuffled_ids = np.random.default_rng(seed).permutation(group_ids)
    group_order = {
        group_id: position
        for position, group_id in enumerate(shuffled_ids)
    }
    result = frame.copy()
    result["_group_order"] = result[GROUP_COLUMN].map(group_order)
    return (
        result.sort_values(
            ["_group_order", "STARTTIME"],
            kind="stable",
        )
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )


def _groups_are_contiguous(groups: pd.Series) -> bool:
    transitions = groups.ne(groups.shift())
    block_counts = groups.loc[transitions].value_counts()
    return bool(not block_counts.empty and block_counts.max() == 1)


def prepare_blood_glucose_dataframe(
    source: pd.DataFrame,
    *,
    shuffle_groups: bool = False,
    group_shuffle_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Return the model table and a non-identifying preparation summary."""
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")

    missing = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in source.columns
    ]
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")

    frame = source.copy()
    for column in [
        "TIMER",
        "STARTTIME",
        "ENDTIME",
        "GLCTIMER",
        "GLCTIMER_AL",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    insulin = (
        frame.loc[frame["INPUT"].notna()]
        .dropna(subset=[GROUP_COLUMN, "STARTTIME"])
        .sort_values([GROUP_COLUMN, "STARTTIME"], kind="stable")
        .reset_index(drop=True)
    )
    glucose = (
        frame.loc[frame["GLC"].notna()]
        .dropna(subset=[GROUP_COLUMN, "GLCTIMER"])
        .sort_values([GROUP_COLUMN, "GLCTIMER"], kind="stable")
        .reset_index(drop=True)
    )

    glucose_by_patient = {
        subject_id: patient_frame
        for subject_id, patient_frame in glucose.groupby(
            GROUP_COLUMN,
            sort=False,
        )
    }
    paired_frames = []
    for subject_id, patient_insulin in insulin.groupby(
        GROUP_COLUMN,
        sort=False,
    ):
        patient_glucose = glucose_by_patient.get(subject_id)
        if patient_glucose is None or patient_glucose.empty:
            continue
        paired_frames.append(
            _attach_next_glucose(patient_insulin, patient_glucose)
        )

    if not paired_frames:
        raise ValueError("No insulin events could be paired with later glucose values.")

    paired = (
        pd.concat(paired_frames, ignore_index=True)
        .dropna(subset=["GLC_next"])
        .reset_index(drop=True)
    )
    paired[TARGET_COLUMN] = (
        paired["GLC_next"]
        .between(70, 180, inclusive="both")
        .astype("int8")
    )

    impossible_glucose = (
        paired["GLC_AL"].notna()
        & paired["GLC_AL"].le(0)
    )
    paired = paired.loc[~impossible_glucose].reset_index(drop=True)

    if shuffle_groups:
        paired = _shuffle_complete_groups(paired, group_shuffle_seed)
    else:
        paired = (
            paired.sort_values(
                [GROUP_COLUMN, "STARTTIME"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    numeric = paired[NUMERIC_PREDICTORS].copy()
    for column in NUMERIC_PREDICTORS:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        ).astype("Float64")
    numeric = numeric.fillna(0.0).astype("float32").reset_index(drop=True)

    encoded = pd.DataFrame(
        {
            "INSULINTYPE_Long": _category_indicator(
                paired["INSULINTYPE"],
                "LONG",
            ),
            "INSULINTYPE_Short": _category_indicator(
                paired["INSULINTYPE"],
                "SHORT",
            ),
            "EVENT_BOLUS_PUSH": _category_indicator(
                paired["EVENT"],
                "BOLUS_PUSH",
            ),
            "EVENT_INFUSION": _category_indicator(
                paired["EVENT"],
                "INFUSION",
            ),
            "GLCSOURCE_AL_FINGERSTICK": _category_indicator(
                paired["GLCSOURCE_AL"],
                "FINGERSTICK",
            ),
            "GLCSOURCE_AL_nan": paired["GLCSOURCE_AL"]
            .isna()
            .astype("float32"),
        }
    ).reset_index(drop=True)

    features = pd.concat([numeric, encoded], axis=1)
    features = (
        features[EXPECTED_FEATURE_COLUMNS]
        .fillna(0.0)
        .astype("float32")
        .reset_index(drop=True)
    )
    target = paired[TARGET_COLUMN].astype("int8").reset_index(drop=True)
    groups = paired[GROUP_COLUMN].copy().reset_index(drop=True)

    if len(features) != len(target) or len(target) != len(groups):
        raise ValueError("X, y, and groups are not aligned.")
    if features.columns.tolist() != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("The prepared feature schema is not the thesis schema.")
    if features.shape[1] != 13:
        raise ValueError("The prepared feature matrix must contain 13 columns.")
    if features.select_dtypes(include=[np.number]).shape[1] != 13:
        raise ValueError("All prepared predictors must be numeric.")
    if features.isna().any().any():
        raise ValueError("The prepared feature matrix contains missing values.")
    if not np.isfinite(features.to_numpy(dtype=np.float32)).all():
        raise ValueError("The prepared feature matrix contains infinite values.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("The prepared target must contain both binary classes.")
    if groups.isna().any():
        raise ValueError("Patient groups contain missing identifiers.")
    if not _groups_are_contiguous(groups):
        raise ValueError("At least one patient occupies multiple row blocks.")

    chronology = paired.groupby(GROUP_COLUMN, sort=False)["STARTTIME"].apply(
        lambda values: values.is_monotonic_increasing
    )
    if not bool(chronology.all()):
        raise ValueError("Within-patient chronological order was not preserved.")

    model_table = features.copy()
    model_table[TARGET_COLUMN] = target
    model_table[GROUP_COLUMN] = groups

    summary = {
        "source_rows": int(len(source)),
        "paired_rows": int(len(model_table)),
        "feature_count": int(features.shape[1]),
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "patient_groups": int(groups.nunique()),
        "class1_prevalence": float(target.mean()),
        "impossible_glc_al_rows_removed": int(impossible_glucose.sum()),
        "group_order_shuffled": bool(shuffle_groups),
        "group_shuffle_seed": int(group_shuffle_seed) if shuffle_groups else None,
        "within_patient_chronology_preserved": True,
    }
    return model_table, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the common 13-feature blood-glucose model table used by "
            "the three thesis scenarios."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the local PhysioNet-derived "
            "LMU_Final_Cleaned_Data.pkl study extract"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "thesis_study_blood_glucose/data/"
            "prepared_blood_glucose_model_table.pkl"
        ),
        help="Destination for the prepared model table",
    )
    parser.add_argument(
        "--shuffle-groups",
        action="store_true",
        help=(
            "Optionally reorder complete patient groups before saving. "
            "The thesis configurations do not require this because V14 "
            "performs seeded group-aware sampling and splitting."
        ),
    )
    parser.add_argument(
        "--group-shuffle-seed",
        type=int,
        default=42,
        help="Seed used only when --shuffle-groups is supplied",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    source = pd.read_pickle(input_path)
    model_table, summary = prepare_blood_glucose_dataframe(
        source,
        shuffle_groups=args.shuffle_groups,
        group_shuffle_seed=args.group_shuffle_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_table.to_pickle(output_path)

    summary_path = output_path.with_suffix(".summary.json")
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared model table: {output_path}")
    print(f"Preparation summary: {summary_path}")
    print(f"Rows: {len(model_table):,}")
    print("Features: 13 encoded numeric predictors")
    print(f"Patient groups: {model_table[GROUP_COLUMN].nunique():,}")
    print(f"Class-1 prevalence: {model_table[TARGET_COLUMN].mean():.6f}")


if __name__ == "__main__":
    main()
