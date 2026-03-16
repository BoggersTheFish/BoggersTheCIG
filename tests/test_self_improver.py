"""Tests for self_improver module."""
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_run_one_cycle_skip_ollama():
    """Run one self-improvement cycle with Ollama skipped (no local server)."""
    from src.self_improver import run_one_cycle

    stats = run_one_cycle(
        skip_ollama=True, skip_git_push=True, skip_tests=True, skip_git_pull=True
    )
    assert "pull_ok" in stats
    assert "coherence_ok" in stats
    assert "tests_ok" in stats
    assert "export_ok" in stats
    assert stats["export_ok"] is True


def test_run_one_cycle_dry_run():
    """Dry run completes without committing."""
    from src.self_improver import run_one_cycle

    stats = run_one_cycle(
        skip_ollama=True, skip_git_push=True, skip_tests=True, skip_git_pull=True, dry_run=True
    )
    assert stats["success"] is True
    assert stats["commit_ok"] is None


def test_ts_coherence_check():
    """TS coherence check runs without error."""
    from src.self_improver import _ts_coherence_check

    ok, coh = _ts_coherence_check()
    assert isinstance(ok, bool)
    assert isinstance(coh, dict)


def test_export_to_obsidian_vault():
    """Export creates files in Obsidian vault."""
    from src.self_improver import _export_to_obsidian_vault
    from src.config import OBSIDIAN_VAULT

    _export_to_obsidian_vault()
    concepts_dir = OBSIDIAN_VAULT / "Concepts"
    assert concepts_dir.exists()


def test_ollama_integration_check():
    """Ollama check returns bool (may be False if Ollama not running)."""
    from ollama_integration import check_ollama_available

    result = check_ollama_available()
    assert isinstance(result, bool)


def test_auto_commit_if_changes_dry_run():
    """auto_commit_if_changes with dry_run returns success and skipped_reason."""
    from src.self_improver import auto_commit_if_changes

    result = auto_commit_if_changes("test-reason", dry_run=True)
    assert "success" in result
    assert result.get("skipped_reason") == "dry-run"
    assert "commit_message" in result
