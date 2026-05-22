# CIG Cleanup Report

Date: 2026-05-22

## Commit

Current cleanup branch:

```text
codex/cig-public-framing-cleanup
```

Local commits in this cleanup pass:

- `51049453 Reframe CIG public documentation`
- `pending`: license, test dependency, and cleanup report follow-up

## Files Changed

- `README.md`
- `LICENSE`
- `requirements-dev.txt`
- `docs/ARCHITECTURE.md`
- `docs/LIMITATIONS.md`
- `docs/FIRST_CONTACT.md`
- `docs/GITHUB_METADATA.md`
- `docs/CLEANUP_REPORT.md`

## Purpose

Reframe BoggersTheCIG as:

```text
Local-first provenance-aware concept graph for confidence-weighted claims, evidence, contradictions, Obsidian-readable memory, and TS-style inspectable knowledge state.
```

The README now states that this is an active experimental prototype, useful as CIG infrastructure, and not a finished autonomous reasoner.

## Test Environment Notes

Initial commands attempted before `requirements-dev.txt` existed:

```bash
python -m pytest
python3 -m pytest
python3 -m unittest discover
```

Observed failures:

- `python` was not found.
- `python3 -m pytest` failed because `pytest` was not installed.
- `python3 -m unittest discover` failed because test modules import `pytest`.

This pass adds `requirements-dev.txt` with minimal pytest runner dependencies. The full runtime dependencies remain in `requirements.txt`.

Follow-up environment checks:

```bash
python3 -m pip install -r requirements-dev.txt
```

Result: failed because the system Python is externally managed under PEP 668.

Temporary venv check:

```bash
python3 -m venv /tmp/boggers_cig_test_venv
/tmp/boggers_cig_test_venv/bin/python -m pip install -r requirements-dev.txt
/tmp/boggers_cig_test_venv/bin/python -m pytest
```

Result after adding `networkx` to `requirements-dev.txt`:

- pytest collected `52` items.
- tests progressed through `test_concept_graph`, `test_e2e`, `test_hardware_adapt`, `test_knowledge_ingest`, `test_language_layer`, and into `test_obsidian_bridge`.
- one visible failure appeared in `tests/test_concept_graph.py` before the run hung later in the Obsidian bridge area.
- the pytest process was manually stopped after it produced no further output for over a minute.

This means the test runner dependency issue is improved, but the suite still needs a clean full dependency/test isolation pass before claiming green tests.

## Push Diagnostics

Earlier push attempts stalled:

```bash
git push -u origin codex/cig-public-framing-cleanup
timeout 180s git push -u origin codex/cig-public-framing-cleanup
```

Symptoms:

- no remote output for several minutes,
- process stayed in pack/remote-transfer,
- second bounded retry timed out.

Repository object state during diagnosis:

- `git count-objects -vH` reported a packed repository around `1.04 GiB`.
- The cleanup commit itself only changed markdown files.
- The largest tracked files are existing memory/provenance and Obsidian snapshot artifacts, not new cleanup files.

Final push command:

```bash
timeout 300s git push --progress origin HEAD:refs/heads/codex/cig-public-framing-cleanup
```

Result: succeeded. The progress output showed the branch upload sending the existing large repository history, around `1.03 GiB`, before creating the remote branch.

## Manual Retry If Needed

If push stalls again, retry from a stable network shell:

```bash
git push --progress origin HEAD:refs/heads/codex/cig-public-framing-cleanup
```

If that still stalls, inspect whether the remote has all main-branch objects and consider pushing from a fresh clone of the remote with the cleanup patch applied.
