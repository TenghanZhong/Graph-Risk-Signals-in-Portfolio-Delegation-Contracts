# Anonymization Notes

This repository is prepared as an anonymous review artifact.

The artifact removes:

- author names and affiliations;
- local Windows user names and absolute local paths;
- local draft metadata in source-file headers;
- private scratch directories and unneeded intermediate outputs.

The repository intentionally keeps stable internal experiment identifiers in
some cached diagnostic CSVs so that results can be traced to the experiment
engine. Paper-facing outputs use anonymous descriptive labels such as
`Feature-only contract`, `Graph-penalty contract`, and `Randomized-score contract`.

Before submission, run:

```bash
python scripts/smoke_test.py
```

The smoke test scans text files for common identity and local-path strings.
