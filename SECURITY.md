# Security and data handling

## Credentials

The pipeline does not contain an API key or token. Optional external services
must obtain credentials from the user's runtime environment. Never place
credentials in Python files, notebooks, JSON configurations, issue reports, or
Git history.

Before committing, inspect staged files:

```bash
git diff --cached
```

If a credential is committed, revoke it immediately and remove it from the full
repository history. Deleting it only from the latest commit is insufficient.

## Research data

Do not commit identifiable, confidential, licensed, or restricted study data.
Use a local path in the JSON configuration and retain only a non-sensitive
dataset identifier, schema description, and checksum where permitted.

Split indices, predictions, group identifiers, and fitted models may also be
sensitive. Review the output directory before sharing any run artifacts.

## Vulnerability reports

Do not include credentials or confidential data in a public issue. Contact the
repository owner privately for security-sensitive reports.

