"""Pytest configuration for isolated, local-only unit tests."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Run before any tests import src.config.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="cig-tests-"))
os.environ.setdefault("TS_DATA_DIR", str(_TEST_ROOT / "data"))
os.environ.setdefault("TS_GRAPHS_DIR", str(_TEST_ROOT / "graphs"))
os.environ.setdefault("TS_MEMORY_DIR", str(_TEST_ROOT / "memory"))
os.environ.setdefault("TS_EVAL_DIR", str(_TEST_ROOT / "eval"))
os.environ.setdefault("TS_VIZ_DIR", str(_TEST_ROOT / "viz"))
os.environ.setdefault("OBSIDIAN_VAULT", str(_TEST_ROOT / "obsidian" / "TS-Knowledge-Vault"))
os.environ.setdefault("GRAPH_URI", "bolt://localhost:7687")
os.environ.setdefault("ENABLE_LARGE_GRAPH", "false")
os.environ.setdefault("ENABLE_EXTERNAL_INGEST", "false")

(_TEST_ROOT / "obsidian" / "TS-Knowledge-Vault" / "Concepts").mkdir(parents=True, exist_ok=True)

import pytest


@pytest.fixture(autouse=True)
def reset_persistent_store_singletons():
    """Keep unit tests from reusing persistent store singletons across cases."""
    yield
    sqlite_mod = sys.modules.get("src.sqlite_store")
    if sqlite_mod is not None and getattr(sqlite_mod, "_default_store", None) is not None:
        try:
            sqlite_mod._default_store.close()
        except Exception:
            pass
        sqlite_mod._default_store = None

    provenance_mod = sys.modules.get("src.provenance_store")
    if provenance_mod is not None:
        provenance_mod._default_store = None


def pytest_sessionfinish(session, exitstatus):
    """Remove the temporary runtime tree created for the test session."""
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
