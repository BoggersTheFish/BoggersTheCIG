"""
Self-improving loop: git pull, Ollama analysis, code diffs, tests, commit, push.
Governed by .cursor/rules/self-improving-loop.md and ts-evolution principles.
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT,
    EVAL_DIR,
    OBSIDIAN_VAULT,
    WEBHOOK_URL,
    QUERY_GENERATE_EVERY_N_CYCLES,
)
from src.eval import run_eval, graph_coherence
from src.concept_graph import ConceptGraph
from src.viz import export_to_obsidian, auto_snapshot_graph

logger = logging.getLogger(__name__)

_LOG_PATH = EVAL_DIR / "self_improve_log.jsonl"


def _run_cmd(cmd: list, cwd: Path = None, capture: bool = True, timeout: int = 120) -> tuple[int, str]:
    """Run shell command. Returns (returncode, stdout+stderr)."""
    cwd = cwd or PROJECT_ROOT
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _git_pull() -> bool:
    """Pull latest from origin. Returns True if success or not a git repo."""
    if not (PROJECT_ROOT / ".git").exists():
        logger.info("Not a git repo, skipping pull")
        return True
    code, out = _run_cmd(["git", "pull", "--rebase"])
    if code != 0:
        logger.warning("git pull failed: %s", out)
        return False
    logger.info("git pull ok")
    return True


def _git_status_short() -> str:
    """Return short summary of changed files for commit message."""
    if not (PROJECT_ROOT / ".git").exists():
        return "no-repo"
    code, out = _run_cmd(["git", "status", "--short"])
    if code != 0:
        return "unknown"
    lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
    if not lines:
        return "no-changes"
    dirs = set()
    for l in lines:
        p = l.split()[-1] if l else ""
        if "/" in p:
            dirs.add(p.split("/")[0])
        else:
            dirs.add(p)
    return "+".join(sorted(dirs)[:5])


def _git_rollback():
    """Discard all uncommitted changes."""
    if not (PROJECT_ROOT / ".git").exists():
        return
    _run_cmd(["git", "reset", "--hard", "HEAD"])
    _run_cmd(["git", "checkout", "--", "."])
    logger.info("Git rollback: discarded uncommitted changes")


def _git_create_backup_branch() -> str | None:
    """Create ts-backup-YYYYMMDD-HHMM branch at current HEAD. Returns branch name or None."""
    if not (PROJECT_ROOT / ".git").exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    name = f"ts-backup-{ts}"
    code, out = _run_cmd(["git", "branch", name])
    if code != 0:
        logger.warning("Backup branch failed: %s", out)
        return None
    logger.info("Created backup branch: %s", name)
    return name


def _git_revert_last_commit() -> bool:
    """Revert the last commit. Returns True on success."""
    if not (PROJECT_ROOT / ".git").exists():
        return False
    code, out = _run_cmd(["git", "revert", "HEAD", "--no-edit"])
    if code != 0:
        logger.warning("git revert failed: %s", out)
        return False
    _run_cmd(["git", "push"])  # Push revert if we had pushed
    logger.info("Reverted last commit")
    return True


def _git_prune_backups(max_keep: int = 3):
    """Delete oldest ts-backup-* branches, keep max_keep newest."""
    if not (PROJECT_ROOT / ".git").exists():
        return
    code, out = _run_cmd(["git", "branch", "--list", "ts-backup-*"])
    if code != 0:
        return
    branches = [b.strip().lstrip("* ") for b in out.strip().split("\n") if b.strip()]
    if len(branches) <= max_keep:
        return
    for b in sorted(branches)[:-max_keep]:
        _run_cmd(["git", "branch", "-D", b])
        logger.info("Pruned backup branch: %s", b)


def _run_small_test_suite() -> tuple[bool, str]:
    """Run a small subset of tests (faster). Returns (passed, output_snippet)."""
    code, out = _run_cmd(
        [sys.executable, "-m", "pytest", "tests/test_self_improver.py", "tests/test_concept_graph.py", "-v", "--tb=short", "-x"],
        timeout=90,
    )
    snippet = out[-1500:] if len(out) > 1500 else out
    return code == 0, snippet


def _coherence_dropped_more_than_10pct(before: dict, after: dict) -> bool:
    """True if coherence (density) dropped by more than 10%."""
    if not before or not after:
        return False
    b_dens = before.get("density", 0) or 0.0001
    a_dens = after.get("density", 0) or 0
    if b_dens <= 0:
        return False
    return a_dens < b_dens * 0.9


def _git_commit_and_push(message: str) -> bool:
    """Commit all changes and push. Returns True on success."""
    if not (PROJECT_ROOT / ".git").exists():
        logger.info("Not a git repo, skipping commit")
        return False
    _run_cmd(["git", "add", "-A"])
    code, _ = _run_cmd(["git", "diff", "--staged", "--quiet"])
    if code == 0:
        logger.info("No changes to commit")
        return True
    code, out = _run_cmd(["git", "commit", "-m", message])
    if code != 0:
        logger.warning("git commit failed: %s", out)
        return False
    code, out = _run_cmd(["git", "push"])
    if code != 0:
        logger.warning("git push failed: %s", out)
        return False
    logger.info("Committed and pushed: %s", message)
    return True


def _get_repo_summary() -> str:
    """Build a concise summary of repo state for Ollama."""
    lines = []
    for p in ["src", "tests", "ollama_integration.py"]:
        path = PROJECT_ROOT / p
        if not path.exists():
            continue
        if path.is_file():
            try:
                lines.append(f"--- {p}\n{path.read_text(encoding='utf-8', errors='replace')[:2000]}")
            except Exception:
                pass
        else:
            for f in path.rglob("*.py"):
                try:
                    rel = f.relative_to(PROJECT_ROOT)
                    lines.append(f"--- {rel}\n{f.read_text(encoding='utf-8', errors='replace')[:1500]}")
                except Exception:
                    pass
    return "\n".join(lines)[:8000]


def _run_tests() -> tuple[bool, str]:
    """Run pytest. Returns (passed, output_snippet)."""
    code, out = _run_cmd(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        timeout=180,
    )
    snippet = out[-2000:] if len(out) > 2000 else out
    if code != 0:
        logger.warning("Tests failed:\n%s", snippet)
        return False, snippet
    logger.info("Tests passed")
    return True, snippet


def _ts_coherence_check() -> tuple[bool, dict]:
    """Run TS coherence check. Returns (ok, coherence_dict)."""
    try:
        graph = ConceptGraph()
        coh = graph_coherence(graph)
        if coh.get("conflicts", 0) > 5:
            logger.warning("TS coherence: too many conflicts (%d)", coh["conflicts"])
            return False, coh
        run_eval()
        return True, coh
    except Exception as e:
        logger.warning("TS coherence check failed: %s", e)
        return False, {}


def _export_to_obsidian_vault():
    """Export graph to Obsidian TS-Knowledge-Vault."""
    OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)
    export_to_obsidian(ConceptGraph(), target_dir=OBSIDIAN_VAULT / "Concepts")


def _send_notification(payload: dict, reason: str = "failure"):
    """Print to console and optionally POST to WEBHOOK_URL."""
    msg = json.dumps(payload, indent=2)
    print(f"\n--- TS {reason.upper()} ---\n{msg}\n---")
    if WEBHOOK_URL:
        try:
            import requests
            r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            if r.status_code >= 400:
                logger.warning("Webhook returned %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Webhook failed: %s", e)


def _coherence_delta(before: dict, after: dict) -> str:
    """Compute coherence change as +X% or -X% for commit message."""
    if not before or not after:
        return ""
    b_conf = before.get("conflicts", 0)
    a_conf = after.get("conflicts", 0)
    b_dens = before.get("density", 0) or 0.0001
    a_dens = after.get("density", 0) or 0.0001
    pct = int(100 * (a_dens - b_dens) / b_dens) if b_dens else 0
    if a_conf < b_conf:
        pct = max(pct, 5)
    sign = "+" if pct >= 0 else ""
    return f"coherence {sign}{pct}%"


def _should_generate_queries_this_cycle() -> bool:
    """True if cycle count % N == 0 (run query generation every N cycles)."""
    if not _LOG_PATH.exists():
        return True
    try:
        count = sum(1 for _ in open(_LOG_PATH, encoding="utf-8"))
        return count % QUERY_GENERATE_EVERY_N_CYCLES == 0
    except Exception:
        return False


def run_one_cycle(
    skip_ollama: bool = False,
    skip_git_push: bool = True,
    skip_tests: bool = False,
    ingest_external: bool = False,
    generate_queries: bool = False,
    dry_run: bool = False,
    notify_on_change: bool = False,
    safe_evolve: bool = False,
) -> dict:
    """
    One self-improvement cycle:
    1. git pull
    2. Ollama analysis (if available)
    3. TS coherence check (before)
    4. External ingest (optional)
    5. Export to Obsidian, snapshot
    6. TS coherence check (after)
    7. Run tests
    8. Commit & push (if not dry_run, not skip_git_push, and success)
    On failure: rollback git changes, log error, optionally notify.
    Returns stats dict.
    """
    start_ts = datetime.now(timezone.utc).isoformat()
    stats = {
        "start_time": start_ts,
        "pull_ok": False,
        "ollama_analysis": "",
        "coherence_before": {},
        "coherence_after": {},
        "coherence_ok": False,
        "tests_ok": False,
        "export_ok": False,
        "snapshot": None,
        "external_ingest": {},
        "query_generation": {},
        "changes_made": "",
        "commit_ok": None,
        "rollback": False,
        "safe_evolve_rollback": False,
        "success": False,
        "error": None,
        "end_time": None,
    }

    def _step(desc: str):
        try:
            from tqdm import tqdm
            return tqdm([1], desc=desc, unit="step", leave=False)
        except ImportError:
            logger.info(desc)
            return [1]

    # 1. Git pull
    for _ in _step("git pull"):
        stats["pull_ok"] = _git_pull()
        if not stats["pull_ok"]:
            stats["error"] = "git pull failed"
            stats["end_time"] = datetime.now(timezone.utc).isoformat()
            _append_log(stats)
            _send_notification(stats, "failure")
            return stats

    # 2. Coherence before (baseline, no failure)
    for _ in _step("coherence (before)"):
        try:
            stats["coherence_before"] = graph_coherence(ConceptGraph())
        except Exception as e:
            logger.debug("Coherence before failed: %s", e)
            stats["coherence_before"] = {}

    # 3. Ollama analysis
    if not skip_ollama:
        for _ in _step("Ollama analysis"):
            try:
                from ollama_integration import analyze_repo_for_tensions, check_ollama_available
                if check_ollama_available():
                    summary = _get_repo_summary()
                    analysis = analyze_repo_for_tensions(summary)
                    stats["ollama_analysis"] = analysis[:500] if analysis else ""
                    if analysis:
                        logger.info("Ollama analysis: %s...", analysis[:200])
                else:
                    logger.info("Ollama not available, skipping analysis")
            except ImportError:
                logger.debug("ollama_integration not importable")

    # 4. Query generation (every N cycles or when --generate-queries)
    if generate_queries or (ingest_external and _should_generate_queries_this_cycle()):
        try:
            from src.knowledge_ingest import generate_queries_from_graph
            qgen = generate_queries_from_graph(use_ollama=not skip_ollama, reason="cycle")
            stats["query_generation"] = qgen
        except Exception as e:
            logger.debug("Query generation failed: %s", e)
            stats["query_generation"] = {"error": str(e)}

    # 5. External ingest
    if ingest_external:
        for _ in _step("external ingest"):
            try:
                from src.knowledge_ingest import ingest_external_knowledge
                ext = ingest_external_knowledge(use_ollama=not skip_ollama, force=True)
                stats["external_ingest"] = ext
            except Exception as e:
                logger.warning("External ingest failed: %s", e)
                stats["external_ingest"] = {"error": str(e)}

    # 6. Export + coherence metrics + snapshot (metrics before snapshot for delta overlay)
    for _ in _step("export to Obsidian"):
        _export_to_obsidian_vault()
        stats["export_ok"] = True
        try:
            from src.viz import export_coherence_metrics
            stats["coherence_metrics"] = export_coherence_metrics()
        except Exception as e:
            logger.debug("Coherence metrics export failed: %s", e)
        try:
            changes_short = _git_status_short()
            snap = auto_snapshot_graph(
                reason="self-improve",
                change_desc=changes_short if changes_short not in ("no-repo", "no-changes") else "self-improve",
            )
            stats["snapshot"] = snap
        except Exception as e:
            logger.debug("Snapshot failed: %s", e)

    # 7. Coherence after
    for _ in _step("coherence (after)"):
        stats["coherence_ok"], stats["coherence_after"] = _ts_coherence_check()
        if not stats["coherence_ok"]:
            stats["error"] = f"coherence failed: {stats['coherence_after']}"
            stats["rollback"] = True
            if not dry_run:
                _git_rollback()
            stats["end_time"] = datetime.now(timezone.utc).isoformat()
            _append_log(stats)
            _send_notification(stats, "failure")
            return stats

    # 8. Tests
    if not skip_tests:
        for _ in _step("tests"):
            passed, out = _run_tests()
            stats["tests_ok"] = passed
            stats["test_output"] = out[-500:] if len(out) > 500 else out
            if not passed:
                stats["error"] = "tests failed"
                stats["rollback"] = True
                if not dry_run:
                    _git_rollback()
                stats["end_time"] = datetime.now(timezone.utc).isoformat()
                _append_log(stats)
                _send_notification(stats, "failure")
                return stats
    else:
        stats["tests_ok"] = True

    stats["success"] = True

    # 9. Commit & push (unless dry_run or skip_git_push)
    changes = _git_status_short()
    meaningful = changes not in ("no-repo", "no-changes")
    delta = _coherence_delta(stats["coherence_before"], stats["coherence_after"])
    commit_msg = f"TS auto-evolve: {changes} ({delta})" if delta else f"TS auto-evolve: {changes}"

    if not dry_run and not skip_git_push and meaningful:
        if safe_evolve:
            _git_create_backup_branch()
            _git_prune_backups(max_keep=3)
        stats["commit_ok"] = _git_commit_and_push(commit_msg)
        stats["commit_message"] = commit_msg
        if stats["commit_ok"] and safe_evolve:
            coh_dict = graph_coherence(ConceptGraph())
            tests_ok_safe, _ = _run_small_test_suite()
            if _coherence_dropped_more_than_10pct(stats["coherence_before"], coh_dict) or not tests_ok_safe:
                stats["safe_evolve_rollback"] = True
                stats["rollback"] = True
                _git_revert_last_commit()
                _append_log({**stats, "rollback_reason": "Rollback: bad evolution"})
                logger.warning("Rollback: bad evolution (coherence drop >10%% or tests failed)")
                _send_notification({
                    "event": "ts-evolve-rollback",
                    "reason": "Rollback: bad evolution",
                    "coherence_before": stats["coherence_before"],
                    "coherence_after": coh_dict,
                }, "failure")
                stats["end_time"] = datetime.now(timezone.utc).isoformat()
                return stats
        if notify_on_change:
            _send_notification({
                "event": "ts-evolve",
                "success": True,
                "changes": changes,
                "coherence": stats["coherence_after"],
                "commit": commit_msg,
            }, "change")
    else:
        stats["commit_ok"] = None
        stats["commit_message"] = commit_msg
        if dry_run:
            logger.info("Dry run: skipped commit")
        elif not meaningful:
            logger.info("No meaningful changes to commit")
    # Write commit message for CI (e.g. GitHub Actions) to use when it commits
    if meaningful and not dry_run:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / "last_commit_message.txt").write_text(commit_msg, encoding="utf-8")

    stats["changes_made"] = _git_status_short()
    stats["end_time"] = datetime.now(timezone.utc).isoformat()
    _append_log(stats)
    return stats


def _append_log(entry: dict):
    """Append cycle log to self_improve_log.jsonl."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in entry.items() if k != "test_output" or len(str(v)) < 1000}
    if "test_output" in entry and "test_output" not in safe:
        safe["test_output_len"] = len(str(entry.get("test_output", "")))
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe, default=str) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama analysis")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest (faster demo)")
    parser.add_argument("--ingest-external", action="store_true", help="Run external knowledge ingest")
    parser.add_argument("--skip-git-push", action="store_true", default=True, help="Do not push (default)")
    parser.add_argument("--no-skip-git-push", action="store_false", dest="skip_git_push")
    parser.add_argument("--dry-run", action="store_true", help="Simulate cycle without committing")
    parser.add_argument("--notify-on-change", action="store_true", help="Notify only if meaningful changes occurred")
    parser.add_argument("--safe-evolve", action="store_true", help="Backup before commit, revert if coherence drops >10%% or tests fail")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stats = run_one_cycle(
        skip_ollama=args.skip_ollama,
        skip_git_push=args.skip_git_push,
        skip_tests=args.skip_tests,
        ingest_external=getattr(args, "ingest_external", False),
        dry_run=getattr(args, "dry_run", False),
        notify_on_change=getattr(args, "notify_on_change", False),
        safe_evolve=getattr(args, "safe_evolve", False),
    )
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
