"""
Subsystem 9: Visualization
Export to Obsidian (Markdown + wikilinks), NetworkX/Matplotlib, graph snapshots.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import VIZ_DIR, OBSIDIAN_VAULT, SNAPSHOTS_DIR, SNAPSHOT_MAX_WIDTH, SNAPSHOT_MAX_HEIGHT
from src.concept_graph import ConceptGraph

logger = logging.getLogger(__name__)


def export_to_obsidian(graph: ConceptGraph = None, subgraph_nodes: list = None, target_dir: Path = None):
    """
    Generate Markdown files per node for Obsidian vault.
    Links via [[wikilinks]]. Uses OBSIDIAN_VAULT/Concepts if set, else VIZ_DIR.
    """
    graph = graph or ConceptGraph()
    out_dir = target_dir or (OBSIDIAN_VAULT / "Concepts")
    out_dir.mkdir(parents=True, exist_ok=True)

    if graph._use_neo4j:
        nodes = [r["name"] for r in graph.cypher("MATCH (n:Concept) RETURN n.name as name")]
    else:
        nodes = list(graph._nx_graph.nodes())

    if subgraph_nodes:
        nodes = [n for n in nodes if n in subgraph_nodes]

    for node in nodes[:500]:  # Limit for large graphs
        neighbors = graph.get_neighbors(node, limit=20)
        content = f"# {node}\n\n"
        content += "## Neighbors\n\n"
        for nb, rel, w in neighbors:
            content += f"- [[{nb}]] ({rel}, weight={w})\n"
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in node)
        path = out_dir / f"{safe_name}.md"
        path.write_text(content, encoding="utf-8")

    logger.info("Exported %d nodes to %s", len(nodes), out_dir)


def plot_networkx(graph: ConceptGraph = None, output_path: Path = None, max_nodes: int = 100):
    """Quick plot via NetworkX + Matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    if graph._use_neo4j:
        # Build subgraph from Cypher
        recs = graph.cypher(
            "MATCH (a:Concept)-[r:RELATES]->(b:Concept) RETURN a.name as a, b.name as b LIMIT $lim",
            {"lim": max_nodes * 2},
        )
        G = nx.DiGraph()
        for r in recs:
            G.add_edge(r["a"], r["b"])
    else:
        G = graph._nx_graph
        if G.number_of_nodes() > max_nodes:
            G = G.subgraph(list(G.nodes())[:max_nodes])

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph, nothing to plot")
        return

    out = output_path or (VIZ_DIR / "graph.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", out)


def _get_delta_from_coherence_log(metrics_dir: Path) -> dict:
    """Read last 2 entries from coherence_log.jsonl, return delta dict."""
    log_path = metrics_dir / "coherence_log.jsonl"
    if not log_path.exists():
        return {}
    try:
        lines = [l.strip() for l in open(log_path, encoding="utf-8") if l.strip()]
        if len(lines) < 2:
            return {}
        prev = json.loads(lines[-2])
        curr = json.loads(lines[-1])
        bn, be = prev.get("nodes", 0), prev.get("edges", 0)
        an, ae = curr.get("nodes", 0), curr.get("edges", 0)
        b_dens = (2 * be / (bn * (bn - 1))) if bn > 1 else 0.0001
        a_dens = (2 * ae / (an * (an - 1))) if an > 1 else 0
        coh_pct = int(100 * (a_dens - b_dens) / b_dens) if b_dens else 0
        return {
            "nodes_delta": an - bn,
            "edges_delta": ae - be,
            "coherence_pct": coh_pct,
        }
    except Exception:
        return {}


def _get_commit_link(project_root: Path) -> str:
    """Return markdown link to latest commit if available."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return ""
        short = r.stdout.strip()
        r2 = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r2.returncode == 0 and r2.stdout.strip():
            url = r2.stdout.strip().replace(".git", "").replace("git@github.com:", "https://github.com/")
            if "github.com" in url:
                return f" [commit]({url}/commit/{short})"
        return f" `{short}`"
    except Exception:
        return ""


def _add_text_overlay_pil(img_path: Path, lines: list[str]):
    """Draw text overlay on PNG using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.debug("Pillow not installed, skipping text overlay")
        return
    try:
        img = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        try:
            import platform
            if platform.system() == "Windows":
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
        y = 8
        for line in lines:
            if line:
                draw.rectangle([(4, y - 2), (img.width - 4, y + 16)], fill=(0, 0, 0, 180))
                draw.text((8, y), line, fill=(255, 255, 255), font=font)
                y += 20
        img.convert("RGB").save(img_path, "PNG")
    except Exception as e:
        logger.debug("PIL overlay failed: %s", e)


def auto_snapshot_graph(
    vault_path: Path = None,
    reason: str = "export",
    delta_metrics: dict = None,
    change_desc: str = None,
) -> str | None:
    """
    Save a visual snapshot of the TS concept graph after vault-modifying operations.
    Uses matplotlib + networkx. Adds Pillow text overlay: timestamp, change desc, delta metrics.
    Saves to vault/snapshots/. Updates snapshots/index.md with clickable image + summary + commit link.
    Returns path to saved PNG or None on failure.
    """
    from src.config import PROJECT_ROOT

    vault = vault_path or OBSIDIAN_VAULT
    snap_dir = vault / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ts_file = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason)[:30]
    fname = f"graph-{ts_file}-{safe_reason}.png"
    out_path = snap_dir / fname

    graph = ConceptGraph()
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.warning("matplotlib not installed, skipping snapshot")
        return None

    if graph._use_neo4j:
        recs = graph.cypher(
            "MATCH (a:Concept)-[r:RELATES]->(b:Concept) RETURN a.name as a, b.name as b LIMIT 500",
            {},
        )
        G = nx.DiGraph()
        for r in recs:
            G.add_edge(r["a"], r["b"])
    else:
        G = graph._nx_graph
        if G.number_of_nodes() > 200:
            G = G.subgraph(list(G.nodes())[:200])

    if G.number_of_nodes() == 0:
        logger.info("Empty graph, skipping snapshot")
        return None

    fig = plt.figure(figsize=(SNAPSHOT_MAX_WIDTH / 100, SNAPSHOT_MAX_HEIGHT / 100), dpi=100)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=6)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()

    delta = delta_metrics or _get_delta_from_coherence_log(vault / "metrics")
    desc = change_desc or reason
    delta_parts = []
    if delta.get("nodes_delta") is not None:
        sign = "+" if delta["nodes_delta"] >= 0 else ""
        delta_parts.append(f"nodes {sign}{delta['nodes_delta']}")
    if delta.get("edges_delta") is not None:
        sign = "+" if delta["edges_delta"] >= 0 else ""
        delta_parts.append(f"edges {sign}{delta['edges_delta']}")
    if delta.get("coherence_pct") is not None:
        sign = "+" if delta["coherence_pct"] >= 0 else ""
        delta_parts.append(f"coherence {sign}{delta['coherence_pct']}%")
    delta_str = ", ".join(delta_parts) if delta_parts else ""

    overlay_lines = [ts_str, f"After: {desc}"]
    if delta_str:
        overlay_lines.append(delta_str)
    _add_text_overlay_pil(out_path, overlay_lines)

    commit_link = _get_commit_link(PROJECT_ROOT)
    _update_snapshot_index(snap_dir, fname, reason, change_desc or reason, delta_str, commit_link)
    logger.info("Saved graph snapshot to %s", out_path)
    return str(out_path)


def _update_snapshot_index(
    snap_dir: Path,
    fname: str,
    reason: str,
    change_summary: str = "",
    delta_str: str = "",
    commit_link: str = "",
):
    """Append entry to snapshots/index.md with clickable image, change summary, commit link."""
    idx_path = snap_dir / "index.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = change_summary or reason
    extra = f" — {delta_str}" if delta_str else ""
    link = f"[![graph]({fname})]({fname})"
    entry = f"\n### {ts}\n\n**After:** {summary}{extra}\n\n{link}{commit_link}\n"
    if idx_path.exists():
        content = idx_path.read_text(encoding="utf-8")
    else:
        content = "# TS Graph Snapshots\n\nAuto-generated after vault-modifying operations.\n"
    content += entry
    idx_path.write_text(content, encoding="utf-8")


def _get_nodes_with_degree(graph: ConceptGraph) -> list[tuple[str, int]]:
    """Return [(node_name, degree), ...] sorted by degree desc."""
    if graph._use_neo4j:
        recs = graph.cypher(
            "MATCH (n:Concept) OPTIONAL MATCH (n)-[r:RELATES]-() WITH n, count(r) as deg RETURN n.name as name, deg ORDER BY deg DESC",
            {},
        )
        return [(r["name"], r["deg"]) for r in recs]
    import networkx as nx
    G = graph._nx_graph
    return sorted(((n, G.degree(n)) for n in G.nodes()), key=lambda x: -x[1])


def export_coherence_metrics(vault_path: Path = None) -> dict:
    """
    Export coherence metrics to vault for Dataview dashboard.
    Appends to metrics/coherence_log.jsonl, writes top_concepts.md, orphans.md,
    recent_commits.md, and generates snapshots/coherence-trend.png.
    Returns metrics dict.
    """
    from src.eval import graph_coherence
    from src.config import PROJECT_ROOT

    vault = vault_path or OBSIDIAN_VAULT
    metrics_dir = vault / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = vault / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    graph = ConceptGraph()
    coh = graph_coherence(graph)
    nodes = coh.get("nodes", 0)
    edges = coh.get("edges", 0)
    conflicts = coh.get("conflicts", 0)
    avg_degree = (2 * edges / nodes) if nodes > 0 else 0

    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "nodes": nodes, "edges": edges, "conflicts": conflicts, "avg_degree": round(avg_degree, 2)}
    log_path = metrics_dir / "coherence_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    nodes_degree = _get_nodes_with_degree(graph)
    top10 = nodes_degree[:10]
    orphans = [(n, d) for n, d in nodes_degree if d < 2]

    (metrics_dir / "top_concepts.md").write_text(
        "# Top Concepts by Link Count\n\n| Concept | Degree |\n|---------|--------|\n"
        + "\n".join(f"| [[{n}]] | {d} |" for n, d in top10),
        encoding="utf-8",
    )
    (metrics_dir / "orphans.md").write_text(
        "# Orphan Concepts (degree < 2)\n\n"
        + "\n".join(f"- [[{n}]] (degree={d})" for n, d in orphans[:100]),
        encoding="utf-8",
    )

    recent_commits = ""
    try:
        import subprocess
        try:
            vault_rel = str(vault.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            vault_rel = "obsidian/"
        r = subprocess.run(
            ["git", "log", "-5", "--oneline", "--", vault_rel],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            recent_commits = r.stdout.strip()
        else:
            r2 = subprocess.run(
                ["git", "log", "-5", "--oneline", "--", "obsidian/"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            recent_commits = r2.stdout.strip() if r2.returncode == 0 else ""
    except Exception:
        pass
    (metrics_dir / "recent_commits.md").write_text(
        "# Recent Commits (Vault)\n\n```\n" + (recent_commits or "No git history") + "\n```",
        encoding="utf-8",
    )

    trend_path = snap_dir / "coherence-trend.png"
    try:
        import matplotlib.pyplot as plt
        lines = [json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip()]
        if len(lines) >= 2:
            xs = list(range(len(lines)))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(xs, [r["nodes"] for r in lines], label="Nodes", marker="o", markersize=3)
            ax.plot(xs, [r["edges"] for r in lines], label="Edges", marker="s", markersize=3)
            ax.plot(xs, [r["conflicts"] for r in lines], label="Conflicts", marker="^", markersize=3)
            ax.set_xlabel("Cycle")
            ax.legend(loc="upper left")
            ax.set_title("TS Coherence Trend")
            plt.tight_layout()
            plt.savefig(trend_path, dpi=100, bbox_inches="tight")
            plt.close()
            logger.info("Saved coherence trend to %s", trend_path)
    except Exception as e:
        logger.debug("Coherence trend chart failed: %s", e)

    logger.info("Exported coherence metrics to %s", metrics_dir)
    return entry
