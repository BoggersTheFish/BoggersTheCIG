# BoggersTheCIG — TS Cognitive Engine

A self-evolving knowledge graph engine built on the **Thinking System (TS)** architecture. It ingests text, extracts concept triples, builds a persistent knowledge graph with epistemic weights, validates hypotheses against real evidence, and autonomously evolves its own knowledge and code — all locally, at $0 cost.

Inspired by ACT-R, Wolfram Alpha's graph reasoning, and DeepMind's symbolic approaches.

---

## What's New (v2)

The engine has been upgraded from a flat triple store to a full **epistemic reasoning system**:

| Capability | Before | After |
|---|---|---|
| Edge metadata | type, weight, source | + confidence, source_type, provenance, created_at, last_reinforced |
| Coherence metric | density + conflict count | + semantic coherence (embedding cosine), avg confidence |
| Hypothesis validation | structural graph check | evidence-based: DuckDuckGo search → semantic similarity |
| Contradiction detection | bidirectional cycle detection | + semantic opposition via embeddings |
| Memory | NetworkX JSON | SQLite (persistent, versioned) + NetworkX cache |
| Forgetting | prune by degree | Ebbinghaus exponential decay; human edits protected |
| Obsidian sync | graph → vault (one-way) | bidirectional: user edits flow back to graph |
| Triple extraction | one static prompt | 5-prompt pool, per-prompt quality tracking, Ollama evolution |
| Commit strategy | binary pass/fail | 3-tier: auto-commit / review branch / stage-only by confidence |
| Code evolution | Ollama suggests (unused) | isolated git worktree, tests + coherence gate, merge if pass |
| API endpoints | /expand, /analyse, /map, /ingest | + /ask, /trace, /contradictions, /bridges, /metrics, /dashboard |

---

## Architecture

The system is a **13-subsystem pipeline** that runs continuously via GitHub Actions (every 5 hours) or locally:

```
Text Input
    │
    ▼
[1] Language Layer         extract_triples_with_confidence() → (S, R, O, confidence)
    │                      Prompt Registry: 5 prompts A/B tested, best auto-selected
    ▼
[2] Concept Graph          SQLite primary store + NetworkX in-memory cache
    │                      Every edge: confidence, source_type, provenance, timestamps
    │                      Ebbinghaus decay: c(t) = c₀·e^(−Δt/stability)
    ▼
[3] Provenance Store       (S,R,O) → source chain; cross-source confidence boosting
    │
    ▼
[4] Core Engine            structural search, pattern discovery, causal chain BFS,
    │                      semantic contradiction detection, bridge node detection
    ▼
[5] Hypothesis Generator   select node → detect gaps → generate candidates
    │                      → validate_with_evidence() via DuckDuckGo + embedding similarity
    │                      → strategy-adaptive selection (least_accessed / high_degree / random)
    ▼
[6] Validation Engine      bias, logic, stability, empirical checks
    ▼
[7] Memory Architecture    SQLite (edges) + Redis (raw queue) + meta rules
    ▼
[8] Self-Improver          3-tier confidence-gated commits
    │                      Ebbinghaus decay per cycle
    │                      Obsidian bidirectional sync
    │                      Code self-modification via isolated git worktree
    │                      Insights.md generation
    ▼
[9] Visualization          Obsidian Markdown export with bridge:: tags, confidence, provenance
    │                      Graph snapshots (Pillow overlays), coherence metrics
    ▼
[10] API                   FastAPI: /ask, /trace, /contradictions, /bridges,
                           /metrics, /dashboard, /expand, /ingest, /analyse
```

---

## Project Structure

```
BoggersTheCIG/
├── src/
│   ├── language_layer.py         # Triple extraction with confidence scores + prompt registry
│   ├── concept_graph.py          # Graph backend (SQLite + NetworkX / Memgraph)
│   ├── core_engine.py            # Reasoning: causal chains, bridge nodes, contradictions
│   ├── hypothesis_generator.py   # Evidence-based hypothesis validation
│   ├── validation_engine.py      # Multi-stage triple validation
│   ├── continuous_thinker.py     # Background reasoning loop (Celery)
│   ├── memory_layers.py          # Memory layer abstraction
│   ├── eval.py                   # Coherence metrics (semantic + structural)
│   ├── viz.py                    # Obsidian export (bridge tags, confidence, provenance)
│   ├── self_improver.py          # Self-improvement cycle with confidence-gated commits
│   ├── knowledge_ingest.py       # External web ingestion via DuckDuckGo
│   ├── obsidian_sync.py          # Bidirectional Obsidian ↔ graph sync
│   ├── obsidian_filesystem_manager.py
│   ├── obsidian_ollama_bridge.py
│   ├── sqlite_store.py           # Persistent SQLite knowledge store
│   ├── provenance_store.py       # Triple source chain + corroboration index
│   ├── prompt_registry.py        # Prompt evolution engine
│   ├── hardware_adapt.py         # Auto-detects hardware, selects Ollama model tier
│   ├── config.py                 # All configuration (env vars)
│   ├── main.py                   # CLI entry point
│   └── api/
│       └── app.py                # FastAPI: all endpoints
├── obsidian/
│   └── TS-Knowledge-Vault/       # Visual memory layer
│       ├── Concepts/             # Node markdown files (bridge-tagged, confidence shown)
│       ├── snapshots/            # Timestamped graph PNGs
│       ├── metrics/              # coherence_log.jsonl
│       ├── Insights.md           # Auto-generated cycle summary
│       ├── Evolution-Log.md      # Full change journal
│       └── TS-Dashboard.md       # Dataview live dashboard
├── memory/
│   ├── knowledge.db              # SQLite knowledge store (primary)
│   ├── provenance_index.json     # Triple → source chain index
│   ├── provenance_store.jsonl    # Provenance audit log
│   └── prompt_registry.json      # Prompt pool + quality scores
├── eval/
│   ├── self_improve_log.jsonl
│   ├── hypothesis_success.json
│   ├── strategy_stats.json       # Per-strategy corroboration rates
│   └── knowledge_ingest.jsonl
├── data/
│   └── queries.json              # Manual + auto-generated search queries
├── graphs/
│   └── networkx_fallback.json    # JSON backup (SQLite is primary)
├── tests/
├── .github/workflows/
│   ├── ts-evolve.yml             # Runs every 5h: self-improve + commit
│   └── ts_graph_builder.yml
├── docker-compose.yaml           # Memgraph + Redis
├── requirements.txt
└── Makefile
```

---

## Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- 8GB+ RAM
- Docker (optional, for Memgraph at >50k nodes)

### Install

```bash
git clone https://github.com/BoggersTheFish/BoggersTheCIG
cd BoggersTheCIG

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Pull an Ollama model (auto-detected based on your hardware)
ollama pull qwen2.5-coder:7b   # 16GB RAM
# or
ollama pull phi3.5-mini-instruct  # 8GB RAM
```

### Hardware auto-detection

The system detects your CPU/RAM/GPU and picks the best model automatically:

| Hardware | Model |
|---|---|
| 8GB RAM, no GPU | phi3.5-mini-instruct |
| 16GB RAM, no GPU | qwen2.5-coder:7b |
| 32GB RAM or 12GB VRAM | qwen2.5-coder:14b |
| 24GB VRAM (RTX 4090) | qwen2.5:32b |
| 48GB VRAM (A100) | llama3.1:70b |

Override: `python src/main.py --force-model llama3.1:8b`

---

## Running

```bash
# Set PYTHONPATH (required)
export PYTHONPATH=.   # Linux/Mac
# set PYTHONPATH=.   # Windows

# Ingest text into the knowledge graph
python src/main.py --input "Gravity bends spacetime"
python src/main.py --input "Explain quantum physics" --no-llm  # rule-based only

# Run one self-improvement cycle
python src/main.py --self-improve

# With external web ingestion
python src/main.py --self-improve --ingest-external

# Safe mode: backup branch, rollback on coherence drop
python src/main.py --self-improve --safe-evolve

# Dry run (no commits)
python src/main.py --self-improve --dry-run

# Analyze your Obsidian vault
python src/main.py --analyze-vault

# Start the API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

Start the server with `uvicorn src.api.app:app --reload`, then:

### Knowledge Query

```
GET /ask?q=why+does+gravity+bend+light
```
Graph-grounded Q&A. Extracts concepts from your question, expands a 2-hop subgraph, serializes it as grounded context, and asks Ollama to answer citing only what's in the graph. Returns the answer, cited node names, and edge count used.

```json
{
  "question": "why does gravity bend light",
  "answer": "Based on the graph: gravity → warps → spacetime (conf: 0.82)...",
  "cited_nodes": ["gravity", "spacetime", "light_path"],
  "context_edges": 12
}
```

### Causal Chain Tracer

```
GET /trace?source=gravity&target=black_hole_formation&max_depth=5
```
Finds a directed causal path through `causes`, `leads_to`, `produces`, `affects`, `influences`, `results_in` edges.

```json
{
  "found": true,
  "path": ["gravity", "mass_concentration", "spacetime_curvature", "black_hole_formation"],
  "relations": ["causes", "leads_to", "causes"],
  "confidence_product": 0.312,
  "explanation": "gravity (causes) → mass_concentration (leads_to) → spacetime_curvature (causes) → black_hole_formation"
}
```

### Contradiction Detection

```
GET /contradictions?sample_size=300
```
Returns both structural (bidirectional causal cycles) and semantic (embedding opposition) contradictions, sorted by severity.

### Bridge Node Detection

```
GET /bridges?top_n=10
```
Nodes that connect otherwise separate semantic clusters. High `bridge_score` = high betweenness centrality + diverse neighbor embeddings.

### Live Metrics

```
GET /metrics
GET /dashboard   ← HTML dashboard, auto-refreshes every 30s
```

### Other

```
GET  /expand/{concept}    Generate hypotheses for a concept
GET  /analyse/{node}      Structural search + conflicts for a node
GET  /map/{domain}        Export domain subgraph to Obsidian
POST /ingest              Add text, extract triples, ingest
GET  /stats               Node/edge counts
```

---

## Self-Improvement Loop

Each cycle does:

1. `git pull`
2. **Ebbinghaus decay** — confidence decays as `c₀ · e^(−Δt/stability)`. Edges not reinforced in 30 days archive below `0.1` confidence. Human-edited edges are decay-protected.
3. **Coherence before** — baseline: semantic coherence, avg confidence, structural density
4. **Obsidian sync** — detect user edits to vault notes, apply as high-confidence (`0.9`) human edges
5. **Ollama analysis** — detect tensions and gaps in the graph
6. **Query generation** (every 3 cycles) — Ollama suggests search queries from low-degree nodes
7. **External ingest** (optional) — DuckDuckGo → triples with provenance URLs
8. **Export to Obsidian** — markdown files with `bridge:: true` tags, confidence, provenance
9. **Insights.md** — cycle summary: top new edges, contradictions, bridge node, corroborated hypotheses
10. **Coherence after** — semantic + structural + confidence check; rollback if any drops >10–15%
11. **Tests** — pytest; rollback on failure
12. **Confidence-gated commit**:
    - `avg_confidence ≥ 0.75` → auto-commit to main
    - `0.5–0.75` → commit to `ts-review-YYYYMMDD-HHMM` branch + webhook notify
    - `< 0.5` → stage only, no commit

### Code Self-Modification (`--safe-evolve`, every 10 cycles)

Ollama suggests improvements to `language_layer.py` or `hypothesis_generator.py`. The patch is applied in an isolated **git worktree**, tests run, coherence checked — merged to main only if both pass.

---

## Confidence System

Every edge in the graph carries a confidence score `[0, 1]`:

| Source | Default confidence |
|---|---|
| Human edit (Obsidian) | 0.9 |
| External web source with URL | 0.7 |
| LLM-extracted | 0.6 |
| Rule-based regex fallback | 0.3 |
| Unverified hypothesis | 0.25 |

**Corroboration boost:** When a second source confirms the same triple, confidence increases by `+0.1` (same source type) or `+0.2` (different source type), capped at `1.0`.

---

## Hypothesis Validation

Hypotheses are no longer accepted on structural grounds alone. Each candidate triple `(A, relation, B)` is tested:

1. Auto-generate DuckDuckGo query: `"A" relation "B"`
2. Extract triples from top 3 results
3. Compute semantic similarity (cosine) between hypothesis and evidence triples
4. Accept if similarity ≥ 0.55 → confidence scales 0.65–0.85 with evidence strength
5. Reject → marked `speculative`, confidence 0.25
6. Outcome recorded in `strategy_stats.json` — the strategy with highest corroboration rate is auto-selected next cycle

---

## Prompt Evolution

The language layer maintains a pool of 5 triple extraction prompts (standard, specificity-focused, chain-of-thought, scientific, compressed). Each prompt tracks its **average confidence of extracted triples**. Every 50 extractions:

- The worst-performing prompt is retired (if pool > 3)
- Ollama generates a variation of the best prompt
- The new prompt joins the pool as a candidate

---

## Obsidian Integration

Open `obsidian/TS-Knowledge-Vault` as an Obsidian vault.

- **Concepts/** — one file per node. Bridge nodes tagged with `bridge:: true` frontmatter. Neighbors show confidence and provenance source.
- **Insights.md** — updated each cycle: top new edges, active contradictions, top bridge node, corroborated hypotheses
- **TS-Dashboard.md** — Dataview queries: coherence trend, top concepts, orphans
- **snapshots/** — timestamped graph PNGs with Pillow overlays
- **Evolution-Log.md** — full change journal

**Bidirectional sync**: When you edit a concept note (add/remove wikilinks), the next self-improve cycle detects the diff and applies your changes to the graph as high-confidence human edges. Your corrections override the LLM.

---

## GitHub Actions

`.github/workflows/ts-evolve.yml` runs every 5 hours:

1. Checkout → install deps → start Ollama
2. `python src/main.py --self-improve --safe-evolve`
3. Coherence assert + tests
4. Confidence-gated commit & push

**Secrets:**
- `GH_PAT` — Personal Access Token with `repo` scope (or use default `GITHUB_TOKEN` with `permissions: contents: write`)

**Trigger manually:** Actions → TS Evolve → Run workflow

---

## Coherence Metrics

`graph_coherence()` now returns:

```json
{
  "nodes": 142,
  "edges": 387,
  "density": 0.0384,
  "conflicts": 2,
  "semantic_coherence": 0.4312,
  "avg_confidence": 0.6841
}
```

The rollback guard in the self-improver checks all three:
- `semantic_coherence` drops > 15% → rollback
- `avg_confidence` drops > 20% → rollback (noise injection detected)
- `density` drops > 10% → rollback (structural collapse)

---

## External Knowledge Ingestion

```bash
# Run with ingestion
python src/main.py --self-improve --ingest-external

# Generate search queries from graph gaps
python src/main.py --generate-queries
```

- Free DuckDuckGo search (no API key), rate-limited 2s between queries
- Each ingested triple gets `source_type="web_search"`, `provenance=<URL>`
- External triples start at confidence `0.7`; boosted on corroboration
- Logs to `eval/knowledge_ingest.jsonl`
- Query auto-generation: Ollama reads low-degree nodes + contradictions, suggests 5–10 DuckDuckGo queries every 3 cycles

---

## Scaling

| Nodes | Backend |
|---|---|
| < 50k | NetworkX (in-memory) + SQLite (persistence) |
| ≥ 50k | Memgraph via Docker (Bolt protocol) |

Large-graph features: community detection sharding, batch semantic search, parallel Ollama instances on different ports.

```bash
# Multi-instance Ollama
ollama serve &
OLLAMA_HOST=0.0.0.0:11435 ollama serve &
python src/main.py --parallel-threads 2 --analyze-vault
```

---

## Testing

```bash
pytest tests/ -v
```

36 tests, 8 skipped (network-dependent). All core subsystems covered.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run from project root; `main.py` adds `sys.path` automatically |
| Push fails in Actions | Add `GH_PAT` secret or set `permissions: contents: write` |
| Ollama 404 | `ollama pull qwen2.5-coder:7b` — bridge uses first available model |
| External ingest skipped | Pass `--ingest-external` flag or `ENABLE_EXTERNAL_INGEST=true` |
| Vault empty | Ensure `Concepts/*.md` files have `# Node` + `- [[X]] (rel, ...)` format |
| DuckDuckGo rate limit | Increase `INGEST_RATE_LIMIT_SEC` (default 2.0s) |
| Low avg_confidence, no commits | Graph has noisy triples; run `--dry-run` to inspect before committing |

---

## Cost

- **Everything**: $0 (local Ollama, DuckDuckGo, SQLite, Docker)
- Optional cloud scaling: $0–$20/month (Neo4j AuraDB free tier, Oracle Free K8s)

---

## License

MIT
