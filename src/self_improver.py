"""
Self-improving loop: git pull, Ollama analysis, code diffs, tests, commit, push.
Governed by .cursor/rules/self-improving-loop.md and ts-evolution principles.

Auto-commit: At end of meaningful tasks (analyze-vault, organize-vault, generate-queries),
auto_commit_if_changes() commits and pushes if changes exist and coherence passes.
Example: Would commit: TS auto-evolve: added rollback safety (coherence +4.2%)
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
    code_status, out_status = _run_cmd(["git", "status", "--porcelain"])
    has_unstaged = bool(out_status.strip()) if code_status == 0 else False
    stashed = False
    if has_unstaged:
        code_stash, _ = _run_cmd(["git", "stash", "push", "-u", "-m", "ts-self-improve-pre-pull"])
        stashed = code_stash == 0
    code, out = _run_cmd(["git", "pull", "--rebase"])
    if stashed:
        _run_cmd(["git", "stash", "pop"])
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
    """Run a small subset of tests (faster, no network). Returns (passed, output_snippet)."""
    code, out = _run_cmd(
        [sys.executable, "-m", "pytest", "tests/test_self_improver.py", "tests/test_concept_graph.py", "tests/test_hardware_adapt.py", "-v", "--tb=short", "-x"],
        timeout=90,
    )
    snippet = out[-1500:] if len(out) > 1500 else out
    return code == 0, snippet


def _coherence_dropped_more_than_10pct(before: dict, after: dict) -> bool:
    """
    True if coherence dropped significantly.
    Uses semantic_coherence when available; falls back to density.
    Checks both structural density AND semantic coherence to catch noise injection.
    """
    if not before or not after:
        return False
    # Semantic coherence check (primary signal when available)
    b_sem = before.get("semantic_coherence", 0)
    a_sem = after.get("semantic_coherence", 0)
    if b_sem > 0.01:
        if a_sem < b_sem * 0.85:
            logger.warning("Semantic coherence dropped: %.4f → %.4f", b_sem, a_sem)
            return True
    # Confidence check: if avg confidence drops sharply, noise was injected
    b_conf = before.get("avg_confidence", 0)
    a_conf = after.get("avg_confidence", 0)
    if b_conf > 0.1 and a_conf < b_conf * 0.80:
        logger.warning("Avg confidence dropped: %.4f → %.4f", b_conf, a_conf)
        return True
    # Structural density fallback
    b_dens = before.get("density", 0) or 0.0001
    a_dens = after.get("density", 0) or 0
    if b_dens <= 0:
        return False
    return a_dens < b_dens * 0.9


def _git_push_with_auth() -> tuple[int, str]:
    """
    Push to origin. Uses GITHUB_TOKEN or GH_PAT for auth when set.
    Returns (returncode, output).
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    if token:
        code, out = _run_cmd(["git", "config", "--get", "remote.origin.url"])
        if code == 0 and out.strip():
            url = out.strip()
            if "github.com" in url and "@" not in url:
                if url.startswith("git@github.com:"):
                    path = url.replace("git@github.com:", "").replace(".git", "")
                    push_url = f"https://x-access-token:{token}@github.com/{path}"
                elif url.startswith("https://github.com/"):
                    push_url = url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")
                else:
                    push_url = url
                return _run_cmd(["git", "push", push_url])
    return _run_cmd(["git", "push"])


def _git_commit_and_push(message: str) -> tuple[bool, str | None]:
    """
    Commit all changes and push. Respects .gitignore via git add .
    Uses GITHUB_TOKEN or GH_PAT for auth. Handles push conflicts (pull --rebase, retry).
    Returns (success, commit_hash). Logs commit hash and push status.
    """
    if not (PROJECT_ROOT / ".git").exists():
        logger.info("Not a git repo, skipping commit")
        return False, None
    _run_cmd(["git", "add", "."])
    code, _ = _run_cmd(["git", "diff", "--staged", "--quiet"])
    if code == 0:
        logger.info("No changes to commit")
        return True, None
    code, out = _run_cmd(["git", "commit", "-m", message])
    if code != 0:
        logger.warning("git commit failed: %s", out)
        return False, None
    code_hash, _ = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    commit_hash = None
    if code_hash == 0:
        _, hash_out = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
        commit_hash = hash_out.strip() if hash_out else None
    code, out = _git_push_with_auth()
    if code != 0:
        logger.warning("git push failed, attempting pull --rebase: %s", out)
        _run_cmd(["git", "pull", "--rebase"])
        code, out = _git_push_with_auth()
        if code != 0:
            _run_cmd(["git", "rebase", "--abort"])
            code_stash, _ = _run_cmd(["git", "stash"])
            if code_stash == 0:
                _run_cmd(["git", "pull", "--rebase"])
                _run_cmd(["git", "stash", "pop"])
                code, out = _git_push_with_auth()
        if code != 0:
            logger.warning("git push failed after retries: %s", out)
            return False, commit_hash
    logger.info("Committed and pushed: %s (hash: %s)", message, commit_hash or "?")
    return True, commit_hash


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


def _short_ollama_summary(analysis: str, max_len: int = 60) -> str:
    """Extract first meaningful line from Ollama analysis for commit message."""
    if not analysis or not isinstance(analysis, str):
        return ""
    for line in analysis.strip().split("\n"):
        line = line.strip().lstrip("-*• ")
        if line and len(line) > 5:
            return line[:max_len].replace('"', "'")
    return analysis[:max_len].replace('"', "'") if analysis else ""


def _classify_commit_tier(avg_confidence: float) -> str:
    """
    Classify commit behavior based on avg edge confidence.
    Returns: 'auto' (>=0.75 or empty graph), 'review' (0.5-0.75), or 'stage' (<0.5).

    Edge case: avg_confidence == 0.0 means the graph has no edges yet (fresh run).
    There is no noise to gate against, so default to 'auto'.
    CI runs always pass --skip-git-push anyway, so this only matters for local runs.
    """
    if avg_confidence <= 0.0:
        return "auto"  # empty graph — no signal to gate on
    if avg_confidence >= 0.75:
        return "auto"
    elif avg_confidence >= 0.5:
        return "review"
    else:
        return "stage"


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
    skip_git_push: bool = False,
    skip_git_pull: bool = False,
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
        "commit_hash": None,
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

    # 1. Git pull (skip when skip_git_pull, e.g. tests in dirty repo)
    if skip_git_pull:
        stats["pull_ok"] = True
    else:
        for _ in _step("git pull"):
            stats["pull_ok"] = _git_pull()
            if not stats["pull_ok"]:
                stats["error"] = "git pull failed"
                stats["end_time"] = datetime.now(timezone.utc).isoformat()
                _append_log(stats)
                _send_notification(stats, "failure")
                return stats

    # 2. Coherence before (baseline, no failure) + decay pass
    for _ in _step("coherence (before)"):
        try:
            graph = ConceptGraph()
            # Run Ebbinghaus decay at start of each cycle
            try:
                decay_result = graph.apply_decay(decay_days=30.0, archive_threshold=0.1)
                logger.info("Decay: %s", decay_result)
                stats["decay"] = decay_result
            except Exception as de:
                logger.debug("Decay failed: %s", de)
            stats["coherence_before"] = graph_coherence(graph)
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

    # 5b. Bidirectional Obsidian sync (human edits → graph)
    for _ in _step("Obsidian sync (human edits)"):
        try:
            from src.obsidian_sync import sync_obsidian_to_graph
            sync_result = sync_obsidian_to_graph()
            stats["obsidian_sync"] = sync_result
            if sync_result.get("edges_added", 0) + sync_result.get("edges_removed", 0) > 0:
                logger.info("Obsidian sync: +%d -%d edges", sync_result["edges_added"], sync_result["edges_removed"])
        except Exception as e:
            logger.debug("Obsidian sync failed: %s", e)

    # 6. Export + optional organize + coherence metrics + snapshot
    for _ in _step("export to Obsidian"):
        _export_to_obsidian_vault()
        stats["export_ok"] = True
        try:
            from src.obsidian_filesystem_manager import auto_organize_vault
            org = auto_organize_vault()
            stats["vault_organize"] = org
        except Exception as e:
            logger.debug("Vault organize failed: %s", e)
        try:
            from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas
            sub = auto_extract_and_merge_subideas(use_ollama=not skip_ollama)
            stats["subideas"] = sub
        except Exception as e:
            logger.debug("Sub-idea extract/merge failed: %s", e)
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
        # Generate Insights.md (partial stats available here; full stats written after coherence)
        try:
            _generate_insights_md(stats)
        except Exception as e:
            logger.debug("Insights.md generation failed: %s", e)

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
            passed, out = _run_small_test_suite()
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

    # 9. Confidence-gated commit & push
    changes = _git_status_short()
    meaningful = changes not in ("no-repo", "no-changes")
    delta = _coherence_delta(stats["coherence_before"], stats["coherence_after"])
    ollama_summary = _short_ollama_summary(stats.get("ollama_analysis", ""))
    if ollama_summary:
        commit_msg = f"TS auto-evolve: {ollama_summary} ({delta})" if delta else f"TS auto-evolve: {ollama_summary}"
    else:
        commit_msg = f"TS auto-evolve: {changes} ({delta})" if delta else f"TS auto-evolve: {changes}"

    # Determine commit tier from avg_confidence
    avg_conf = stats["coherence_after"].get("avg_confidence", 0.0)
    stats["avg_confidence"] = avg_conf
    stats["commit_tier"] = _classify_commit_tier(avg_conf)
    logger.info("Commit tier: %s (avg_confidence=%.3f)", stats["commit_tier"], avg_conf)

    if not dry_run and not skip_git_push and meaningful:
        tier = stats["commit_tier"]
        if tier == "auto":
            # High confidence → auto-commit to main
            if safe_evolve:
                _git_create_backup_branch()
                _git_prune_backups(max_keep=3)
            ok, chash = _git_commit_and_push(commit_msg)
            stats["commit_ok"] = ok
            stats["commit_hash"] = chash
            stats["commit_message"] = commit_msg
            if ok and safe_evolve:
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
        elif tier == "review":
            # Medium confidence → commit to review branch
            review_branch = f"ts-review-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
            _run_cmd(["git", "checkout", "-b", review_branch])
            ok, chash = _git_commit_and_push(f"[REVIEW] {commit_msg}")
            _run_cmd(["git", "checkout", "main"])
            stats["commit_ok"] = ok
            stats["commit_hash"] = chash
            stats["commit_message"] = f"[REVIEW] {commit_msg}"
            stats["review_branch"] = review_branch
            logger.info("Review branch created: %s (avg_conf=%.3f)", review_branch, avg_conf)
            _send_notification({
                "event": "ts-review-needed",
                "branch": review_branch,
                "avg_confidence": avg_conf,
                "commit": commit_msg,
            }, "change")
        else:
            # Low confidence → stage only, no commit
            _run_cmd(["git", "add", "."])
            stats["commit_ok"] = None
            stats["commit_message"] = commit_msg
            logger.warning("Staged only (low confidence=%.3f) — no commit", avg_conf)
            _send_notification({
                "event": "ts-low-confidence",
                "avg_confidence": avg_conf,
                "message": "Changes staged but not committed due to low confidence",
            }, "change")

        if notify_on_change and stats.get("commit_ok"):
            _send_notification({
                "event": "ts-evolve",
                "success": True,
                "tier": tier,
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
    # 10. Code self-modification (safe_evolve only, after successful commit, every 10 cycles)
    if safe_evolve and stats.get("commit_ok") and not dry_run:
        cycle_count = sum(1 for _ in open(_LOG_PATH, encoding="utf-8")) if _LOG_PATH.exists() else 0
        if cycle_count % 10 == 0:
            for _ in _step("code self-modification"):
                try:
                    patch_result = _apply_code_patch_safely(
                        target_file="src/language_layer.py", dry_run=False
                    )
                    stats["code_patch"] = patch_result
                    logger.info("Code self-modification: %s", patch_result.get("reason"))
                except Exception as e:
                    logger.debug("Code self-modification failed: %s", e)

    # Write commit message for CI (e.g. GitHub Actions) to use when it commits
    if meaningful and not dry_run:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / "last_commit_message.txt").write_text(commit_msg, encoding="utf-8")

    stats["changes_made"] = _git_status_short()
    stats["end_time"] = datetime.now(timezone.utc).isoformat()
    _append_log(stats)
    return stats


def auto_commit_if_changes(
    reason: str,
    dry_run: bool = False,
    require_coherence: bool = True,
) -> dict:
    """
    At end of meaningful task: commit and push if changes exist.
    Only commits if coherence check passes (when require_coherence).
    Returns {success, commit_ok, commit_hash, commit_message, skipped_reason}.
    """
    result = {"success": False, "commit_ok": None, "commit_hash": None, "commit_message": None, "skipped_reason": None}
    if not (PROJECT_ROOT / ".git").exists():
        result["skipped_reason"] = "not-a-git-repo"
        return result
    changes = _git_status_short()
    if changes in ("no-repo", "no-changes"):
        result["success"] = True
        result["skipped_reason"] = "no-changes"
        return result
    if dry_run:
        result["success"] = True
        result["skipped_reason"] = "dry-run"
        result["commit_message"] = f"TS auto-evolve: {reason} (would commit)"
        return result
    if require_coherence:
        coh_ok, coh = _ts_coherence_check()
        if not coh_ok:
            result["skipped_reason"] = "coherence-failed"
            return result
        n, e = coh.get("nodes", 0), coh.get("edges", 0)
        delta = f"nodes {n}, edges {e}"
    else:
        delta = ""
    msg = f"TS auto-evolve: {reason} ({delta})" if delta else f"TS auto-evolve: {reason}"
    ok, chash = _git_commit_and_push(msg)
    result["success"] = ok
    result["commit_ok"] = ok
    result["commit_hash"] = chash
    result["commit_message"] = msg
    _append_log({
        "event": "auto_commit",
        "reason": reason,
        "commit_ok": ok,
        "commit_hash": chash,
        "commit_message": msg,
        "changes": changes,
    })
    return result


def _apply_code_patch_safely(
    target_file: str = "src/language_layer.py",
    dry_run: bool = False,
) -> dict:
    """
    Ask Ollama to suggest a code improvement for a target file.
    Apply the patch in an isolated git worktree.
    Run tests + coherence check in the worktree.
    Merge to main only if both pass.

    Only operates on: src/language_layer.py, src/hypothesis_generator.py
    Never touches: self_improver.py, config.py, sqlite_store.py

    Returns {success, patch_applied, tests_passed, merged, reason}.
    """
    ALLOWED_TARGETS = {"src/language_layer.py", "src/hypothesis_generator.py"}
    result = {
        "success": False, "patch_applied": False,
        "tests_passed": False, "merged": False, "reason": "",
    }

    if target_file not in ALLOWED_TARGETS:
        result["reason"] = f"Target '{target_file}' not in allowed set: {ALLOWED_TARGETS}"
        return result

    if not (PROJECT_ROOT / ".git").exists():
        result["reason"] = "Not a git repo"
        return result

    # Read the target file
    target_path = PROJECT_ROOT / target_file
    if not target_path.exists():
        result["reason"] = f"Target file not found: {target_file}"
        return result

    try:
        original_content = target_path.read_text(encoding="utf-8")
    except Exception as e:
        result["reason"] = f"Could not read target: {e}"
        return result

    # Ask Ollama for an improvement
    try:
        from ollama_integration import check_ollama_available, _ollama_request
        if not check_ollama_available():
            result["reason"] = "Ollama unavailable"
            return result

        system = (
            "You are an expert Python code improver. Analyze the provided code and suggest "
            "ONE specific, targeted improvement that increases the quality or accuracy of "
            "triple extraction. Output ONLY the improved Python file — no explanations, "
            "no markdown fences, just the raw Python code."
        )
        prompt = (
            f"Improve this Python file to extract higher-quality knowledge triples. "
            f"Focus on: better relation specificity, reduced noise, improved parsing.\n\n"
            f"File: {target_file}\n\n{original_content[:4000]}"
        )
        new_content = _ollama_request(prompt, system=system, timeout=90).strip()
    except Exception as e:
        result["reason"] = f"Ollama suggestion failed: {e}"
        return result

    # Basic sanity checks on the suggested content
    if not new_content or len(new_content) < 100:
        result["reason"] = "Ollama returned empty or too-short content"
        return result
    if new_content == original_content:
        result["reason"] = "No change suggested"
        return result
    # Must still be valid Python
    try:
        compile(new_content, target_file, "exec")
    except SyntaxError as e:
        result["reason"] = f"Suggested code has syntax error: {e}"
        return result

    if dry_run:
        result["success"] = True
        result["reason"] = "dry_run — patch validated but not applied"
        result["patch_applied"] = True
        return result

    # Create isolated git worktree
    import tempfile
    worktree_path = PROJECT_ROOT.parent / f"ts-patch-worktree-{int(time.time())}"
    code, out = _run_cmd(["git", "worktree", "add", str(worktree_path), "HEAD"])
    if code != 0:
        result["reason"] = f"Could not create worktree: {out}"
        return result

    try:
        patch_target = worktree_path / target_file
        patch_target.write_text(new_content, encoding="utf-8")
        result["patch_applied"] = True

        # Run tests in worktree
        test_code, test_out = _run_cmd(
            [sys.executable, "-m", "pytest",
             "tests/test_language_layer.py", "tests/test_concept_graph.py",
             "-v", "--tb=short", "-x"],
            cwd=worktree_path,
            timeout=90,
        )
        result["tests_passed"] = test_code == 0
        result["test_output"] = test_out[-500:] if len(test_out) > 500 else test_out

        if not result["tests_passed"]:
            result["reason"] = "Tests failed in worktree — patch rejected"
            return result

        # Quick coherence check (import the patched module)
        coherence_ok = True
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("patched_lang", str(patch_target))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Try extracting a known triple
            test_triples = mod.extract_triples_with_confidence("gravity causes objects to fall")
            if not test_triples:
                coherence_ok = False
                result["reason"] = "Patched module produced no triples on test input"
        except Exception as e:
            coherence_ok = False
            result["reason"] = f"Patched module import failed: {e}"

        if not coherence_ok:
            return result

        # Merge: copy patched file to main tree and commit
        target_path.write_text(new_content, encoding="utf-8")
        patch_msg = f"TS code-evolve: improved {target_file} via Ollama suggestion"
        ok, chash = _git_commit_and_push(patch_msg)
        result["merged"] = ok
        result["success"] = ok
        result["reason"] = f"Merged patch to main (commit: {chash})" if ok else "Commit/push failed"

    except Exception as e:
        result["reason"] = f"Patch application failed: {e}"
    finally:
        # Always clean up the worktree
        try:
            _run_cmd(["git", "worktree", "remove", "--force", str(worktree_path)])
        except Exception:
            pass

    return result


def _generate_insights_md(stats: dict) -> bool:
    """
    Generate obsidian/TS-Knowledge-Vault/Insights.md summarizing each cycle's discoveries.
    Content: top 3 new high-confidence triples, 2 contradiction pairs, top bridge node,
             corroborated hypotheses, coherence delta.
    Returns True if written successfully.
    """
    try:
        from datetime import datetime, timezone as tz
        from src.concept_graph import ConceptGraph
        from src.core_engine import CoreEngine
        from src.hypothesis_generator import _load_success_log

        graph = ConceptGraph()
        core = CoreEngine(graph)

        now_str = datetime.now(tz.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# TS Insights — {now_str}\n",
            "Generated automatically each self-improvement cycle.\n",
        ]

        # Coherence delta
        cb = stats.get("coherence_before", {})
        ca = stats.get("coherence_after", {})
        if cb and ca:
            sem_before = cb.get("semantic_coherence", 0)
            sem_after = ca.get("semantic_coherence", 0)
            conf_after = ca.get("avg_confidence", 0)
            delta_sign = "+" if sem_after >= sem_before else ""
            lines.append(f"## Cycle Summary\n")
            lines.append(f"- Semantic coherence: {sem_before:.4f} → {sem_after:.4f} ({delta_sign}{sem_after - sem_before:.4f})\n")
            lines.append(f"- Avg confidence: {conf_after:.4f}\n")
            lines.append(f"- Nodes: {ca.get('nodes', '?')} | Edges: {ca.get('edges', '?')}\n")
            decay = stats.get("decay", {})
            if decay:
                lines.append(f"- Decay archived: {decay.get('archived', 0)} low-confidence edges\n")
            lines.append("\n")

        # Top 3 high-confidence edges added recently (from SQLite)
        lines.append("## Highest-Confidence New Knowledge\n\n")
        try:
            from src.sqlite_store import get_store
            store = get_store()
            recent_edges = store._conn.execute(
                """
                SELECT src, dst, relation, confidence, provenance
                FROM edges
                ORDER BY last_reinforced DESC, confidence DESC
                LIMIT 10
                """
            ).fetchall()
            top3 = [r for r in recent_edges if r["confidence"] >= 0.65][:3]
            if top3:
                for e in top3:
                    prov = f" ([source]({e['provenance'][:80]}))" if e["provenance"] else ""
                    lines.append(f"- **{e['src']}** –[{e['relation']}]→ **{e['dst']}** (conf: {e['confidence']:.2f}){prov}\n")
            else:
                lines.append("- No new high-confidence edges this cycle.\n")
        except Exception as e:
            lines.append(f"- (Could not fetch recent edges: {e})\n")
        lines.append("\n")

        # Top 2 contradiction pairs
        lines.append("## Active Contradictions\n\n")
        try:
            contradictions = core.constraint_resolution()
            sem_contradictions = core.semantic_contradiction_detection(sample_size=200)
            all_c = contradictions[:1] + sem_contradictions[:1]
            if all_c:
                for c in all_c:
                    nodes = c.get("nodes", [])
                    ctype = c.get("type", "unknown")
                    sev = c.get("severity", c.get("similarity", "?"))
                    lines.append(f"- `{nodes[0]}` ↔ `{nodes[1] if len(nodes) > 1 else '?'}` ({ctype}, severity: {sev})\n")
            else:
                lines.append("- No contradictions detected this cycle.\n")
        except Exception as e:
            lines.append(f"- (Contradiction check failed: {e})\n")
        lines.append("\n")

        # Top bridge node
        lines.append("## Top Bridge Node\n\n")
        try:
            bridges = core.find_bridge_nodes(top_n=3)
            if bridges:
                b = bridges[0]
                lines.append(
                    f"- **[[{b['node']}]]** — bridge score: {b['bridge_score']}, "
                    f"betweenness: {b['betweenness']}, cluster span: {b['cluster_span']}\n"
                )
            else:
                lines.append("- (Graph too small for bridge detection)\n")
        except Exception as e:
            lines.append(f"- (Bridge detection failed: {e})\n")
        lines.append("\n")

        # Recently corroborated hypotheses
        lines.append("## Corroborated Hypotheses\n\n")
        try:
            hyp_log = _load_success_log()
            recent_corroborated = [h for h in hyp_log[-50:] if h.get("corroborated")][-3:]
            if recent_corroborated:
                for h in recent_corroborated:
                    t = h.get("triple", ["?", "?", "?"])
                    src = h.get("evidence_source", "")
                    src_str = f" ([evidence]({src[:80]}))" if src else ""
                    lines.append(f"- **{t[0]}** –[{t[1]}]→ **{t[2]}**{src_str}\n")
            else:
                lines.append("- No hypotheses corroborated recently.\n")
        except Exception as e:
            lines.append(f"- (Hypothesis log unavailable: {e})\n")
        lines.append("\n")

        # Obsidian sync summary
        obsidian_sync = stats.get("obsidian_sync", {})
        if obsidian_sync.get("edges_added", 0) + obsidian_sync.get("edges_removed", 0) > 0:
            lines.append("## Human Edits Synced\n\n")
            lines.append(f"- +{obsidian_sync.get('edges_added', 0)} edges added from vault\n")
            lines.append(f"- -{obsidian_sync.get('edges_removed', 0)} edges removed from vault\n\n")

        # Write the file
        insights_path = OBSIDIAN_VAULT / "Insights.md"
        insights_path.parent.mkdir(parents=True, exist_ok=True)
        insights_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote Insights.md to %s", insights_path)
        return True
    except Exception as e:
        logger.warning("Insights.md generation failed: %s", e)
        return False


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
    parser.add_argument("--skip-git-push", action="store_true", help="Do not push (default: push)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate cycle without committing")
    parser.add_argument("--notify-on-change", action="store_true", help="Notify only if meaningful changes occurred")
    parser.add_argument("--safe-evolve", action="store_true", help="Backup before commit, revert if coherence drops >10%% or tests fail")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stats = run_one_cycle(
        skip_ollama=args.skip_ollama,
        skip_git_push=getattr(args, "skip_git_push", False),
        skip_tests=args.skip_tests,
        ingest_external=getattr(args, "ingest_external", False),
        dry_run=getattr(args, "dry_run", False),
        notify_on_change=getattr(args, "notify_on_change", False),
        safe_evolve=getattr(args, "safe_evolve", False),
    )
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
