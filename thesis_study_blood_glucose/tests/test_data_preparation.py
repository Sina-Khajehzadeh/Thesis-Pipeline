import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STUDY_ROOT.parent
MODULE_PATH = STUDY_ROOT / "prepare_blood_glucose_data.py"

spec = importlib.util.spec_from_file_location(
    "prepare_blood_glucose_data",
    MODULE_PATH,
)
preparation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preparation)

loader_spec = importlib.util.spec_from_file_location(
    "dataset_loader",
    REPOSITORY_ROOT / "dataset_loader.py",
)
dataset_loader = importlib.util.module_from_spec(loader_spec)
assert loader_spec.loader is not None
loader_spec.loader.exec_module(dataset_loader)


def synthetic_source() -> pd.DataFrame:
    rows = []
    categories = [
        (1, "Long", "BOLUS_PUSH", "FINGERSTICK", 100.0),
        (2, "Short", "INFUSION", None, 220.0),
        (3, "Basal", "OTHER", "LAB", 150.0),
        (4, "Long", "INFUSION", "FINGERSTICK", 60.0),
    ]
    for subject_id, insulin_type, event, source, future_glucose in categories:
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=subject_id)
        rows.append({
            "SUBJECT_ID": subject_id,
            "STARTTIME": base + pd.Timedelta(hours=1),
            "GLCTIMER": pd.NaT,
            "GLCTIMER_AL": base,
            "GLC": np.nan,
            "GLCSOURCE": None,
            "LOS_ICU_days": subject_id,
            "first_ICU_stay": subject_id % 2 == 0,
            "INPUT": float(subject_id),
            "INPUT_HRS": 1.5,
            "INFXSTOP": 0,
            "GLC_AL": np.nan if subject_id == 2 else 100 + subject_id,
            "RULE": subject_id,
            "INSULINTYPE": insulin_type,
            "EVENT": event,
            "GLCSOURCE_AL": source,
        })
        rows.append({
            "SUBJECT_ID": subject_id,
            "STARTTIME": pd.NaT,
            "GLCTIMER": base + pd.Timedelta(hours=2),
            "GLCTIMER_AL": base + pd.Timedelta(hours=2),
            "GLC": future_glucose,
            "GLCSOURCE": "LAB",
            "LOS_ICU_days": subject_id,
            "first_ICU_stay": subject_id % 2 == 0,
            "INPUT": np.nan,
            "INPUT_HRS": np.nan,
            "INFXSTOP": np.nan,
            "GLC_AL": np.nan,
            "RULE": np.nan,
            "INSULINTYPE": None,
            "EVENT": None,
            "GLCSOURCE_AL": None,
        })
    return pd.DataFrame(rows)


class BloodGlucosePreparationTest(unittest.TestCase):
    def test_preparation_produces_the_thesis_schema(self):
        table, summary = preparation.prepare_blood_glucose_dataframe(
            synthetic_source()
        )

        expected_columns = [
            *preparation.EXPECTED_FEATURE_COLUMNS,
            preparation.TARGET_COLUMN,
            preparation.GROUP_COLUMN,
        ]
        self.assertEqual(table.columns.tolist(), expected_columns)
        self.assertEqual(table.shape, (4, 15))
        self.assertEqual(summary["feature_count"], 13)
        self.assertEqual(summary["patient_groups"], 4)

        features = table[preparation.EXPECTED_FEATURE_COLUMNS]
        self.assertTrue(
            all(str(dtype) == "float32" for dtype in features.dtypes)
        )
        self.assertFalse(features.isna().any().any())
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertEqual(set(table[preparation.TARGET_COLUMN]), {0, 1})

    def test_category_indicators_and_zero_fill(self):
        table, _ = preparation.prepare_blood_glucose_dataframe(
            synthetic_source()
        )
        by_patient = table.set_index(preparation.GROUP_COLUMN)

        self.assertEqual(by_patient.loc[1, "INSULINTYPE_Long"], 1.0)
        self.assertEqual(by_patient.loc[2, "INSULINTYPE_Short"], 1.0)
        self.assertEqual(by_patient.loc[1, "EVENT_BOLUS_PUSH"], 1.0)
        self.assertEqual(by_patient.loc[2, "EVENT_INFUSION"], 1.0)
        self.assertEqual(
            by_patient.loc[1, "GLCSOURCE_AL_FINGERSTICK"],
            1.0,
        )
        self.assertEqual(by_patient.loc[2, "GLCSOURCE_AL_nan"], 1.0)
        self.assertEqual(by_patient.loc[2, "GLC_AL"], 0.0)

    def test_optional_group_shuffle_is_deterministic_and_keeps_blocks(self):
        source = synthetic_source()
        first, first_summary = preparation.prepare_blood_glucose_dataframe(
            source,
            shuffle_groups=True,
            group_shuffle_seed=42,
        )
        second, _ = preparation.prepare_blood_glucose_dataframe(
            source,
            shuffle_groups=True,
            group_shuffle_seed=42,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first_summary["group_order_shuffled"])
        groups = first[preparation.GROUP_COLUMN]
        transitions = groups.ne(groups.shift())
        self.assertEqual(groups.loc[transitions].value_counts().max(), 1)

    def test_prepared_table_hands_off_to_the_v14_dataset_loader(self):
        table, _ = preparation.prepare_blood_glucose_dataframe(
            synthetic_source()
        )
        features, target, groups, info = dataset_loader.prepare_dataset(
            {
                "dataframe": table,
                "target": preparation.TARGET_COLUMN,
                "positive_label": 1,
                "group_column": preparation.GROUP_COLUMN,
                "feature_columns": preparation.EXPECTED_FEATURE_COLUMNS,
                "drop_constant": False,
                "keep_dataframe": True,
            },
            verbose=False,
        )

        self.assertEqual(features.shape, (4, 13))
        self.assertEqual(features.columns.tolist(), preparation.EXPECTED_FEATURE_COLUMNS)
        self.assertEqual(set(target), {0, 1})
        self.assertEqual(len(groups), 4)
        self.assertEqual(info["n_groups"], 4)


if __name__ == "__main__":
    unittest.main()
