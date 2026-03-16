# Full-TS Cognitive Architecture

A complete, working full-scale **Thinking System (TS)** cognitive architecture, evolving from the GOAT-TS (Graph Of All Thoughts - Thinking System) concept. A "living knowledge engine" that ingests text, builds concept graphs, activates ideas, decays noise, spots tensions, and generates hypotheses—inspired by ACT-R, Wolfram Alpha's graph reasoning, and DeepMind's symbolic approaches.

## Features

- **Language Layer**: Extract concept triples (Subject, Relation, Object) from text via LLM
- **Concept Graph**: Memgraph/Neo4j or NetworkX fallback for knowledge representation
- **Core Reasoning**: Structural search, pattern discovery, compression, constraint resolution
- **Hypothesis Generator**: Gap detection, candidate generation, self-improving prompts
- **Validation Engine**: Logical consistency, bias checks, empirical compatibility
- **Continuous Loop**: Celery + Redis for background thinking tasks
- **Memory Architecture**: Raw → Structured → Hypothesis → Meta layers
- **Ethical Safeguards**: Hardcoded rules, bias filtering, harmful content rejection

## Project Structure

```
full-ts-cognitive-architecture/
├── src/                    # Core code
│   ├── language_layer.py    # Triple extraction from text
│   ├── concept_graph.py     # Graph DB operations
│   ├── core_engine.py       # Reasoning engine
│   ├── hypothesis_generator.py
│   ├── validation_engine.py
│   ├── continuous_thinker.py
│   ├── memory_layers.py
│   ├── viz.py              # Obsidian/NetworkX export
│   ├── self_improve.py
│   ├── data_pipeline.py
│   ├── eval.py
│   ├── deploy.py
│   ├── main.py             # Entry point
│   └── api/
│       └── app.py          # FastAPI endpoints
├── data/                   # Input data
│   ├── inputs/
│   └── raw/
├── graphs/                 # DB exports
├── models/                 # LLM checkpoints
├── memory/                 # Memory layer storage
│   ├── raw/
│   ├── structured/
│   ├── hypothesis/
│   └── meta/
├── eval/                   # Benchmarks and logs
├── viz/                    # Obsidian vault output
├── api/                    # (duplicate for clarity)
├── tests/                  # Pytest
├── docker-compose.yaml
├── requirements.txt
├── setup.py
└── Makefile
```

## Setup

### Prerequisites

- Python 3.12+
- 8GB+ RAM (laptop)
- Docker & Docker Compose (for Memgraph/Neo4j)
- Optional: Free cloud GPU (Colab, Kaggle, Lightning AI) for heavy LLM work

### Installation

```bash
# Clone or create project
cd full-ts-cognitive-architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start graph DB (Memgraph) via Docker
docker-compose up -d

# Start Redis (for Celery)
docker-compose up -d redis

# Initialize graph schema
python -c "from src.concept_graph import ConceptGraph; ConceptGraph().init_schema()"
```

### Hardware Check

```bash
python src/deploy.py --check
```

Suggests free cloud if RAM < 8GB or no GPU detected.

## Run Commands

```bash
# Set PYTHONPATH (required for module resolution)
# Linux/Mac: export PYTHONPATH=.
# Windows:   set PYTHONPATH=.

# Single input (one-shot reasoning)
python src/main.py --input "Gravity bends spacetime"
python src/main.py --input "Explain quantum physics" --no-llm  # rule-based only

# Ingest text into graph
python src/main.py --input "Gravity bends spacetime" --ingest-only

# Continuous thinking loop (background)
celery -A src.tasks worker --loglevel=info &
python src/main.py --mode continuous --sleep 60
# Monitor eval/continuous_log.jsonl for hypothesis generation/validation cycles

# API server
uvicorn src.api.app:app --reload --host 0.0.0.0

# View graph in Obsidian: open obsidian/TS-Knowledge-Vault as vault → Graph View shows clusters
```

## Self-Improving Loop

The TS engine can evolve itself using Cursor + Obsidian + Ollama (local). No paid APIs.

### Prerequisites

- **Ollama** installed and running (`ollama serve`)
- Model: `ollama pull qwen2.5-coder:7b` or `ollama pull llama3.1:8b`
- **Obsidian** with Dataview plugin (for TS-Dashboard queries)

### How to Start the Self-Improving Loop

**1. One-time setup**

```bash
# Install pre-commit hook (TS coherence check before each commit)
scripts/install_pre_commit.sh   # Linux/Mac
scripts/install_pre_commit.bat  # Windows

# Pull Ollama model (if using)
ollama pull qwen2.5-coder:7b
```

**2. Run one self-improvement cycle**

```bash
# From project root
python src/main.py --self-improve

# Dry run: simulate without committing (safe to test)
python src/main.py --self-improve --dry-run

# Notify only when meaningful changes occur (requires WEBHOOK_URL)
python src/main.py --self-improve --notify-on-change

# Safe evolve: backup before commit, revert if coherence drops >10% or tests fail
python src/main.py --self-improve --no-skip-git-push --safe-evolve

# Or via script
scripts/run_self_improvement.sh      # Linux/Mac
scripts/run_self_improvement.bat     # Windows
```

**3. Run continuous loop with evolution**

```bash
# Thinking loop + full self-improver every 5 iterations
python src/main.py --mode continuous --sleep 60 --evolve-every 5
```

**4. Obsidian as visual memory**

- Open `obsidian/TS-Knowledge-Vault` as an Obsidian vault
- Graph View shows concept clusters (Concepts folder)
- **TS-Dashboard.md** — Coherence dashboard with Dataview/DataviewJS:
  - Nodes/edges/contradictions over time (from `metrics/coherence_log.jsonl`)
  - Top 10 most-linked concepts, orphan concepts (degree < 2)
  - Recent commits affecting vault, coherence trend chart (matplotlib PNG)
  - Auto-refreshes when Dataview re-evaluates

**5. GitHub Actions (headless evolution)**

- `.github/workflows/ts-evolve.yml` runs every 6 hours
- Installs Ollama, runs `--self-improve --skip-tests`, commits & pushes changes
- **Secrets setup**: Settings → Secrets and variables → Actions → New repository secret
  - `GH_PAT` (optional): Personal Access Token with `repo` scope for push from forks
  - Without GH_PAT: uses default `GITHUB_TOKEN` (requires `permissions: contents: write`)
- **Trigger**: Actions → TS Evolve → Run workflow
- **Failure notification**: On failure, logs to workflow summary (GitHub Actions UI)
- **Commit message**: Uses `TS auto-evolve: [changes] (coherence +X%)` from `eval/last_commit_message.txt`

**5b. Safe evolution (--safe-evolve)**

When committing, `--safe-evolve` adds a rollback safety net:

1. **Before commit**: Create backup branch `ts-backup-YYYYMMDD-HHMM` at current HEAD
2. **After commit**: Run coherence check + small test suite (self_improver, concept_graph)
3. **If coherence drops >10%** (density) **or tests fail**: `git revert` last commit, push revert, log "Rollback: bad evolution"
4. **Backups**: Keep max 3 `ts-backup-*` branches; delete oldest when creating new

Use with `--no-skip-git-push` to enable commits:

```bash
python src/main.py --self-improve --no-skip-git-push --safe-evolve
```

**6. External knowledge ingestion**

```bash
# With --ingest-external flag (runs even if ENABLE_EXTERNAL_INGEST=false)
python src/main.py --self-improve --ingest-external

# Or enable via env for persistent use
ENABLE_EXTERNAL_INGEST=true python src/main.py --self-improve --ingest-external
```

- Uses DuckDuckGo (free), extracts triples via Ollama, ingests into graph
- Rate-limited 2s between queries. Edit `data/queries.json` for search terms
- Logs to `eval/knowledge_ingest.jsonl` with timestamp + source

**6b. Self-generate search queries from graph gaps**

```bash
# Standalone: generate queries from low-degree nodes + contradictions
python src/main.py --generate-queries

# With self-improve (runs every N cycles when --ingest-external)
python src/main.py --self-improve --ingest-external --generate-queries
```

- Uses Ollama to suggest 5–10 high-value DuckDuckGo queries from graph gaps
- Saves to `data/queries.json` with `manual` (preserved) + `generated` + `last_generated` (timestamp, reason)
- Runs automatically every `QUERY_GENERATE_EVERY_N_CYCLES` (default 3) when `--ingest-external` is used
- Fallback: uses manual queries when Ollama unavailable. $0 cost (local Ollama only)

**7. Analyze Obsidian vault**

```bash
python src/main.py --analyze-vault
python src/main.py --analyze-vault --no-ollama   # Use rule-based only (no Ollama)
```

- Reads vault recursively, parses neighbor format `[[X]] (rel, weight=N)`
- Chunks large notes (1500 chars, 200 overlap), sends to Ollama for concept extraction
- Detects contradictions via core_engine
- Progress logging every 20 files. Limit: `VAULT_MAX_FILES` (default 200)

**8. Graph snapshots**

- Auto-saved after: knowledge ingest, vault analysis, self-improve cycle, single-run export
- Location: `obsidian/TS-Knowledge-Vault/snapshots/`
- Format: `graph-YYYY-MM-DD-HH-MM-<reason>.png`
- **Text overlay** (Pillow): timestamp, "After: [change description]", delta metrics (nodes +X, edges +Y, coherence +Z%)
- **Index** (`snapshots/index.md`): clickable image, change summary, link to commit (when in git)
- Size: Resized to 1920×1080 or smaller (configurable via `SNAPSHOT_MAX_WIDTH`, `SNAPSHOT_MAX_HEIGHT`)
- View: Open `snapshots/index.md` in Obsidian or any markdown viewer to browse graph evolution

**9. Notifications (optional)**

- Set `WEBHOOK_URL` (Slack, Discord, or custom webhook) to receive alerts
- **On failure**: Always prints to console + POSTs to webhook (error, rollback status)
- **On change** (with `--notify-on-change`): POSTs only when meaningful changes were committed

```bash
# Slack incoming webhook
export WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Discord webhook
export WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Run with notifications
python src/main.py --self-improve --no-skip-git-push --notify-on-change
```

**Obsidian plugin combo** (optional two-way sync): Text Extractor + Obsidian Advanced URI

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `python src/main.py` (main.py adds `sys.path` automatically) |
| GitHub Actions: push fails | Add `GH_PAT` secret (PAT with `repo` scope) or ensure `permissions: contents: write` |
| Ollama 404 / model not found | `ollama pull qwen2.5-coder:7b` or `ollama pull llama3.1:8b`. Bridge uses first available model if configured model missing |
| External ingest returns "skipped" | Pass `--ingest-external` (forces ingest) or set `ENABLE_EXTERNAL_INGEST=true` |
| Vault empty / no triples | Ensure `obsidian/TS-Knowledge-Vault/Concepts/` has `.md` files with `# Node` and `- [[X]] (rel, ...)` |
| DuckDuckGo rate limit | Increase `INGEST_RATE_LIMIT_SEC` (default 2.0) |
| WEBHOOK_URL not firing | Check URL is valid; Slack/Discord expect JSON body |

### What the Loop Does

1. **git pull** — Get latest code (tqdm progress)
2. **Coherence (before)** — Baseline metrics for commit message
3. **Ollama analysis** — Detect tensions, gaps, performance issues (if Ollama running)
4. **External ingest** — (optional) Search web, extract triples, ingest (tqdm)
5. **Export to Obsidian** — Concepts → Markdown with wikilinks + graph snapshot
6. **Coherence (after)** — Fail and rollback if conflicts > 5
7. **Run tests** — pytest; on failure: rollback, log, notify
8. **Commit** — `TS auto-evolve: [changes] (coherence +X%)` on success; rollback on failure
9. **Log** — `eval/self_improve_log.jsonl` (start/end time, coherence before/after, changes, test results)

## Free Resource Rotation

| Resource | Free Option | Limits |
|----------|-------------|--------|
| Graph DB | Memgraph (Docker) | Local, unlimited |
| Graph DB Cloud | Neo4j AuraDB Free | 50k nodes |
| LLM | Hugging Face (Llama-3.2-3B) | Local/Colab |
| Vector Search | FAISS (CPU) | Local |
| Cache/Queue | Redis (Docker) | Local |
| Cloud GPU | Colab, Kaggle | Session limits |
| K8s | Minikube, Oracle Free | Limited |

## Cost Estimate

- **Core (local)**: $0
- **Cloud scaling**: $0–$20/month (optional free tiers)

## Scaling to Billions

- **Sharding**: Memgraph clusters, graph partitioning
- **Vector DB**: Pinecone free tier or FAISS distributed
- **Celery**: Distributed workers, Redis Cluster
- **K8s**: Minikube local → Oracle Free Tier for production

## Testing

```bash
pytest tests/ -v
```

## Simulated First Run (Logs / Commits on First Run)

**GitHub Actions (ts-evolve) — first run:**
```
Checkout ✓
Set up Python 3.12 ✓
Install dependencies ✓
Install Ollama ✓ (or continue-on-error)
Run self-improvement cycle ✓
  - git pull (or skip if not repo)
  - TS coherence check
  - Export to Obsidian
Run TS coherence check ✓
Run tests ✓ (or continue-on-error)
Commit and push: "No changes to commit" (or commit if eval/graphs/obsidian changed)
```
If any step fails: "Notify on failure" writes to workflow summary.

**External ingest (--ingest-external):**
```
ensure_queries_file() → data/queries.json exists
For each query: sleep(2), DuckDuckGo search, extract triples (Ollama or rule-based)
Validate, filter harmful, ingest
Log: eval/knowledge_ingest.jsonl
```

**Analyze vault:**
```
Analyzing vault: N markdown files
Processing file 1/N: Gravity.md
...
{files_read, triples_ingested, contradictions}
```

## Simulated First Run (What Happens)

**GitHub Actions (ts-evolve):**
1. Checkout repo
2. Install Python 3.12 + deps
3. Install Ollama, pull tinyllama (or skip on error)
4. Run `python src/main.py --self-improve --skip-tests` → coherence check, export to Obsidian, tests skipped
5. TS coherence assert (conflicts ≤ 10)
6. pytest (continue-on-error)
7. If any files changed (eval/, graphs/, obsidian/): commit + push

**External ingest (ENABLE_EXTERNAL_INGEST=true):**
1. Load queries from data/queries.json
2. DuckDuckGo search per query (rate-limited 2s)
3. Ollama extracts triples from snippets (or rule-based fallback)
4. Validate, filter harmful, ingest into graph
5. Log to eval/knowledge_ingest.jsonl

**Analyze vault:**
1. Read obsidian/TS-Knowledge-Vault/*.md
2. Parse neighbor format: `[[X]] (rel, weight=N)` → (subject, rel, X)
3. For non-neighbor content: chunk, send to Ollama (or rule-based)
4. Ingest triples, detect contradictions
5. If triples_ingested > 0: auto_snapshot_graph(reason="after-analyze-vault")
6. Returns {files_read, triples_ingested, contradictions}

**Graph snapshot (after vault-modifying ops):**
1. Load ConceptGraph, build NetworkX subgraph (≤200 nodes)
2. Render via matplotlib, save to snapshots/graph-{timestamp}-{reason}.png
3. Append entry to snapshots/index.md with link and summary

## License

MIT
