# Testing

Use `python3` on this repo. Some systems do not provide a `python` command.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The dev requirements are intentionally small. They cover the default unit test path without installing the full ML stack from `requirements.txt`.

## Default Unit Tests

```bash
python -m pytest -m "not slow and not external and not integration"
```

Default tests are expected to use temporary runtime paths. They should not mutate tracked files under `memory/`, `graphs/`, `obsidian/`, `.obsidian/`, `data/`, `eval/`, or `viz/`.

## Full Test Run

```bash
python -m pytest
```

The full suite includes tests marked `slow`, `external`, or `integration`. Those tests may require optional runtime services or heavier dependencies, such as local Ollama, Obsidian/vault workflows, plotting dependencies, or broader self-improvement workflow checks.

## Isolation Fixed In This Pass

- `tests/conftest.py` now routes test data, graph, memory, eval, viz, and Obsidian vault paths into a temporary directory before `src.config` is imported.
- SQLite and provenance store defaults now derive from `src.config.MEMORY_DIR`, so test path overrides do not write to tracked `memory/` files.
- Persistent store singletons are reset between tests.
- Self-improver tests are marked `integration`.
- Visualization tests are marked `slow` and use `tmp_path` where they write files.

## Notes

If a test writes to a tracked runtime file, treat that as a test isolation bug. Do not commit generated database, vault, snapshot, or provenance changes as part of routine test runs.
