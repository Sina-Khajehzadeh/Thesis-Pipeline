import copy
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = ROOT / "thesis_study_blood_glucose"
CONFIG_ROOT = STUDY_ROOT / "configs"

SCENARIO_FILES = {
    "no_budget": CONFIG_ROOT / "scenario_1_no_budget.json",
    "tabpfn_budget": CONFIG_ROOT / "scenario_2_tabpfn_budget.json",
    "xgboost_budget": CONFIG_ROOT / "scenario_3_xgboost_fixed_029.json",
}

EXPECTED_MODELS = [
    "TabPFN",
    "L-SLR",
    "Augmented_SLR",
    "RandomForest",
    "XGBoost",
    "CatBoost",
]

EXPECTED_SAMPLE_SIZES = [
    1000,
    3000,
    10000,
    15000,
    25000,
    35000,
    40000,
    46000,
    50000,
    60000,
    86000,
    "full",
]


def deep_merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_with_extends(path: Path, seen=None) -> dict:
    path = path.resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"Configuration cycle: {path}")
    seen.add(path)

    current = json.loads(path.read_text(encoding="utf-8"))
    parents = current.pop("extends", None)
    parents = [] if parents is None else (
        parents if isinstance(parents, list) else [parents]
    )
    result = {}
    for parent in parents:
        result = deep_merge(
            result,
            load_with_extends(path.parent / parent, seen),
        )
    return deep_merge(result, current)


class ThesisScenarioConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            name: load_with_extends(path)
            for name, path in SCENARIO_FILES.items()
        }

    def test_common_data_and_experiment_settings_are_identical(self):
        comparable_keys = [
            "iterations",
            "base_seed",
            "sample_sizes",
            "data",
            "train_frac",
            "inner_validation_frac",
            "sampling",
            "splitting",
            "preprocessing",
            "optuna",
            "cpu_parallelism",
        ]
        reference = self.configs["no_budget"]
        for name, config in self.configs.items():
            for key in comparable_keys:
                self.assertEqual(
                    config[key],
                    reference[key],
                    f"{name} differs for common key {key}",
                )

            self.assertEqual(config["iterations"], 20)
            self.assertEqual(config["base_seed"], 2025)
            self.assertEqual(config["sample_sizes"], EXPECTED_SAMPLE_SIZES)
            self.assertEqual(
                config["sampling"],
                {"strategy": "original_prevalence"},
            )
            self.assertEqual(
                config["splitting"]["strategy"],
                "stratified_group",
            )
            self.assertTrue(config["splitting"]["group_aware"])
            self.assertTrue(config["splitting"]["strict"])
            self.assertTrue(config["splitting"]["require_groups"])
            self.assertEqual(
                config["preprocessing"]["mode"],
                "preprocessed_numeric",
            )
            self.assertEqual(len(config["data"]["feature_columns"]), 13)

    def test_no_budget_scenario(self):
        config = self.configs["no_budget"]
        scenario = config["scenarios"]["No_Budgeting"]
        self.assertFalse(scenario["budgeting"]["enabled"])
        self.assertIsNone(scenario["budget_reference_model"])
        self.assertEqual(scenario["enabled_models"], EXPECTED_MODELS)
        tabpfn = scenario["model_overrides"]["TabPFN"]
        self.assertEqual(tabpfn["execution"]["path"], "cloud")
        self.assertEqual(tabpfn["model_path"], "v3_default")

    def test_tabpfn_runtime_budget_scenario(self):
        config = self.configs["tabpfn_budget"]
        scenario = config["scenarios"]["TabPFN_Runtime_Budget"]
        self.assertEqual(scenario["budget_reference_model"], "TabPFN")
        self.assertTrue(scenario["budgeting"]["enabled"])
        self.assertEqual(
            scenario["budgeting"]["applies_to"],
            "optuna_tuning_only",
        )
        self.assertEqual(
            scenario["budgeting"]["non_tuned_reference_runtime_basis"],
            "reference_execution_runtime",
        )
        self.assertEqual(scenario["enabled_models"], EXPECTED_MODELS)
        self.assertEqual(
            scenario["model_overrides"]["TabPFN"]["execution"]["path"],
            "cloud",
        )

    def test_xgboost_fixed_context_scenario(self):
        config = self.configs["xgboost_budget"]
        scenario = config["scenarios"][
            "XGBoost_Runtime_Budget_29pct_Context"
        ]
        self.assertEqual(scenario["budget_reference_model"], "XGBoost")
        self.assertTrue(scenario["budgeting"]["enabled"])
        self.assertEqual(
            scenario["budgeting"]["applies_to"],
            "optuna_tuning_only",
        )
        self.assertEqual(scenario["enabled_models"], EXPECTED_MODELS)

        overrides = scenario["model_overrides"]
        tabpfn = overrides["TabPFN"]
        self.assertEqual(tabpfn["execution"]["path"], "local")
        self.assertEqual(tabpfn["execution"]["local_device"], "cpu")
        context = tabpfn["local_tabpfn_budget"]["context_strategy"]
        self.assertEqual(
            context,
            {
                "strategy": "fixed",
                "fraction": 0.29,
                "fraction_denominator": "total_sample",
            },
        )
        for model_name in EXPECTED_MODELS[1:]:
            self.assertEqual(
                overrides[model_name]["execution"]["device"],
                "cpu",
            )

        raw_text = SCENARIO_FILES["xgboost_budget"].read_text(
            encoding="utf-8"
        )
        self.assertIsNone(re.search(r"\b1[.]65\b|\b1[.]85\b", raw_text))

    def test_no_fixed_expected_hash_is_embedded(self):
        preparation_source = (
            STUDY_ROOT / "prepare_blood_glucose_data.py"
        ).read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(
                r"(?im)^\s*expected_(?:sha256|hash)\s*=",
                preparation_source,
            )
        )


if __name__ == "__main__":
    unittest.main()
