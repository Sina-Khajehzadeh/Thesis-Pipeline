# Contributing

Changes should preserve the distinction between scientific methodology and
software maintenance.

## Before changing the pipeline

- Open an issue describing the purpose and expected effect.
- State whether the change affects models, search spaces, splitting,
  preprocessing, metrics, runtime budgets, energy accounting, or TabPFN
  context behavior.
- Add or update the relevant configuration and field documentation when
  configuration behavior changes.
- Do not commit data, credentials, checkpoints, or generated experiment runs.

## Validation

At minimum, run:

```bash
python -m py_compile V14_Thesis_Pipeline.py dataset_loader.py
python V14_Thesis_Pipeline.py --self-test
```

Changes affecting execution should also use a dry run, a small synthetic
end-to-end run, `--validate-run`, and saved-data plot regeneration.

## Pull requests

Keep pull requests focused. Document:

- the files changed;
- whether scientific behavior changed;
- the configuration used for testing; and
- the validation result.
