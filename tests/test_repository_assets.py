import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryAssetsTest(unittest.TestCase):
    def test_json_documents_and_config_parents(self):
        paths = [
            ROOT / "configs" / "base_config.example.json",
            ROOT / "docs" / "V14_CONFIGURATION_DICTIONARY.json",
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

    def test_configuration_dictionary(self):
        path = ROOT / "docs" / "V14_CONFIGURATION_DICTIONARY.json"
        dictionary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(dictionary["pipeline_version"], "V14")

        fields = [
            field
            for section in dictionary["sections"]
            for field in section["fields"]
        ]
        paths = [field["path"] for field in fields]
        self.assertEqual(len(paths), len(set(paths)))
        required_keys = {
            "path",
            "type",
            "required",
            "default",
            "allowed",
            "description",
            "example",
        }
        for field in fields:
            self.assertTrue(required_keys <= set(field), field.get("path"))

    def test_readme_local_links(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        targets = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
        for target in targets:
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local_path = target.split("#", 1)[0]
            self.assertTrue((ROOT / local_path).exists(), local_path)

    def test_notebooks_have_no_saved_errors(self):
        for name in [
            "V14_Thesis_Pipeline_Reader_Guide.ipynb",
            "V14_Configuration_Dictionary.ipynb",
        ]:
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
        for name in ["V14_Thesis_Pipeline.py", "dataset_loader.py"]:
            source = (ROOT / name).read_text(encoding="utf-8")
            compile(source, name, "exec")


if __name__ == "__main__":
    unittest.main()
