import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryAssetsTest(unittest.TestCase):
    def test_json_documents_and_config_parents(self):
        paths = [
            ROOT / "configs" / "base_config.example.json",
            *sorted((ROOT / "thesis_study_blood_glucose" / "configs").glob("*.json")),
        ]
        for path in paths:
            self.assertTrue(path.is_file(), f"Missing JSON document: {path}")
            document = json.loads(path.read_text(encoding="utf-8"))
            parents = document.get("extends", [])
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                self.assertTrue(
                    (path.parent / parent).resolve().is_file(),
                    f"Missing configuration parent: {parent}",
                )

    def test_readme_local_links(self):
        readmes = [
            ROOT / "README.md",
            ROOT / "thesis_study_blood_glucose" / "README.md",
            ROOT / "thesis_study_blood_glucose" / "DATA_DICTIONARY.md",
        ]
        for readme in readmes:
            text = readme.read_text(encoding="utf-8")
            targets = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
            for target in targets:
                target = target.strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                local_path = target.split("#", 1)[0]
                self.assertTrue((readme.parent / local_path).exists(), local_path)

    def test_reader_assets_do_not_reference_removed_files(self):
        removed_names = {
            "V14_dataset_template.example.json",
            "scenario_xgboost_budget.example.json",
            "DATA_DICTIONARY_TEMPLATE.md",
            "PUBLICATION_MANIFEST.md",
            "RELEASE_CHECKLIST.md",
            "CITATION.cff.template",
            "V14_Configuration_Dictionary.ipynb",
            "V14_CONFIGURATION_DICTIONARY.json",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
        }
        reader_assets = [
            ROOT / "README.md",
            ROOT / "V14_Thesis_Pipeline_Reader_Guide.ipynb",
            ROOT / "thesis_study_blood_glucose" / "README.md",
        ]
        for asset in reader_assets:
            text = asset.read_text(encoding="utf-8")
            for name in removed_names:
                self.assertNotIn(name, text, f"{asset.name}: {name}")

    def test_workflow_runs_study_validation(self):
        workflow = (ROOT / ".github" / "workflows" / "validate.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("thesis_study_blood_glucose/tests", workflow)

    def test_notebooks_have_no_saved_errors(self):
        for name in ["V14_Thesis_Pipeline_Reader_Guide.ipynb"]:
            notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            errors = [
                output
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]
            self.assertFalse(errors, name)

    def test_primary_python_sources_compile(self):
        for name in [
            "V14_Thesis_Pipeline.py",
            "dataset_loader.py",
            "thesis_study_blood_glucose/prepare_blood_glucose_data.py",
        ]:
            source = (ROOT / name).read_text(encoding="utf-8")
            compile(source, name, "exec")


if __name__ == "__main__":
    unittest.main()
