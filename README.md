# discovery-evals

Reproducible benchmarks and verifiers for computational chemistry and
structure-based modelling, with pinned public inputs.

## What this is

Each benchmark is a self-contained task: a written specification, a
programmatic verifier, and inputs pinned by identifier and checksum so a
run is reproducible from a clean machine. A task passes or fails on the
verifier, not on inspection.

## Directory contract

```
tasks/<id>/
  spec.md        Intent / Prompt + assumptions / Verifier
  prompt.txt     the session's first user turn, verbatim
  verify.py      the task's verifier
  notes.md       provenance, licences, retrieval dates, checksums
  env.yaml       pinned per-task environment
  config/        pinned inputs: frozen data and parameters
  refs/          citations: DOIs, licences, fetch notes

tests/           harness-level tests only
```

A task's verifier lives in that task's own directory, because it ships as
part of the task. Root `tests/` covers shared repo tooling and nothing else.

`config/` and `refs/` are separate because they are different kinds of thing.
`config/` is data the task is run on, pinned and checksummed. `refs/` points at
sources outside the tree and holds no data: a source that may not be
redistributed is cited there by DOI, and one that may be is still cited rather
than bundled unless the task needs the file itself.

## Environments

`uv` is the only environment manager.

```
uv sync            # repo tooling (pytest, ruff) only
uv run pytest
uv run ruff check .
```

The repo root declares no runtime dependencies. Scientific packages are
declared and pinned per task in that task's `env.yaml`, and are not
installed speculatively at the root.

## License

MIT. All dependencies are permissively licensed; no GPL code enters this tree.
