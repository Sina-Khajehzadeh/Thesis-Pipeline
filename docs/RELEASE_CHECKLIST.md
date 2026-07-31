# Public repository release checklist

## Owner decisions

- [ ] Select a software license and add the corresponding `LICENSE` file.
- [ ] Complete `CITATION.cff.template`, rename it to `CITATION.cff`, and add
      the author's full citation details and ORCID if applicable.
- [ ] Add the thesis title, institution, year, and permanent repository link.
- [x] Keep archived V12/V13 development files out of the public repository.
- [ ] Review `docs/PUBLICATION_MANIFEST.md` and confirm the intended file set.

## Data and security

- [ ] Confirm that no raw or derived confidential data are tracked.
- [ ] Confirm that no API key, token, `.env` file, credential, or private URL is
      present in the working tree or Git history.
- [ ] Review predictions, group identifiers, split indices, fitted models, and
      figures for disclosure risk.
- [ ] Test `.gitignore` before the first public push.

## Reproducibility

- [ ] Record the definitive Python and package versions.
- [ ] Archive the final JSON configuration used in the thesis.
- [ ] Record the V14 and `dataset_loader.py` SHA256 checksums.
- [ ] Run compilation and the V14 self-tests.
- [ ] Run a small dry run and end-to-end example.
- [ ] Confirm that `--validate-run` passes.
- [ ] Confirm that plots regenerate from saved numerical artifacts.

## Documentation

- [ ] Check every README link on GitHub.
- [ ] Execute both notebooks from a clean environment.
- [ ] Replace all placeholder paths and citation fields.
- [x] Add a short changelog or release note for the public version.
- [ ] Create a tagged release and archive it with a DOI service if required.
