# BoggersTheCIG

Local-first provenance-aware concept graph for confidence-weighted claims, evidence, contradictions, Obsidian-readable memory, and TS-style inspectable knowledge state.

**Status:** active experimental prototype. Useful as CIG infrastructure; not a finished autonomous reasoner.

**Canonical route:** [TS-Start-Here](https://github.com/BoggersTheFish/TS-Start-Here) -> [TS-Reasoner-v0](https://github.com/BoggersTheFish/TS-Reasoner-v0) -> [TensionLM](https://github.com/BoggersTheFish/TensionLM) -> TS-Codex-OS / TS-Core / CIG.

## What It Is

BoggersTheCIG is a local-first claim/evidence graph. It ingests text, extracts concept triples, stores provenance, tracks confidence, surfaces possible contradictions, and can export graph state into an Obsidian-readable vault.

The repo is part of the broader BoggersTheFish TS / Thinking System stack. In that stack, CIG means provenance-aware claim graph infrastructure: claims, sources, confidence, contradictions, bridge nodes, and inspection surfaces.

## What It Is Not

- Not the whole TS system.
- Not a finished autonomous reasoner.
- Not a production knowledge base.
- Not a formal theorem prover.
- Not proof that confidence scores are calibrated truth probabilities.
- Not safe to run as an unsupervised self-modifying agent.

Experimental self-improvement and code-modification paths require human review.

## Core Features

- Text ingestion into subject/relation/object triples.
- Local concept graph backed by SQLite and NetworkX cache paths.
- Provenance records for claim sources and reinforcement.
- Confidence-weighted edges.
- Heuristic contradiction detection.
- Bridge-node and causal-chain inspection.
- Obsidian vault export for human-readable memory.
- FastAPI endpoints for graph inspection and query paths.
- Optional Ollama/local LLM extraction where available.

## Architecture

```text
text input
  -> triple extraction
  -> concept graph
  -> provenance/confidence store
  -> contradiction and bridge checks
  -> Obsidian export / API / visualization
```

Main modules:

- `src/language_layer.py`: triple extraction and optional local model prompts.
- `src/concept_graph.py`: graph backend and edge metadata.
- `src/provenance_store.py`: source-chain and corroboration records.
- `src/core_engine.py`: graph search, bridge detection, and contradiction signals.
- `src/knowledge_ingest.py`: external ingestion helpers.
- `src/obsidian_sync.py`: Obsidian graph export/sync.
- `src/api/app.py`: FastAPI inspection endpoints.
- `src/self_improver.py`: experimental confidence-gated self-improvement loop.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

```bash
git clone https://github.com/BoggersTheFish/BoggersTheCIG
cd BoggersTheCIG

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .

export PYTHONPATH=.
python src/main.py --input "Gravity bends spacetime"
```

Optional local LLM paths require Ollama and a local model:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
python src/main.py --input "Gravity bends spacetime"
```

Rule-based mode:

```bash
python src/main.py --input "Gravity bends spacetime" --no-llm
```

API:

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Minimal Example

```bash
export PYTHONPATH=.
python src/main.py --input "Alice supports the claim that gravity affects light."
python src/main.py --input "Bob disputes the claim that gravity affects light."
```

Then inspect generated graph state in:

- `memory/`
- `graphs/`
- `obsidian/TS-Knowledge-Vault/`

Exact outputs depend on whether rule-based extraction or local LLM extraction is enabled.

## Repo Map

```text
src/
  language_layer.py              triple extraction
  concept_graph.py               graph backend
  provenance_store.py            provenance records
  core_engine.py                 graph search, contradictions, bridges
  knowledge_ingest.py            external ingestion helpers
  obsidian_sync.py               Obsidian sync/export
  api/app.py                     FastAPI layer
  self_improver.py               experimental self-improvement path

tests/                           pytest suite
memory/                          local knowledge/provenance stores
graphs/                          graph exports/cache
obsidian/TS-Knowledge-Vault/     Obsidian-readable memory
docs/                            architecture, limits, first-contact docs
```

## Experimental Features

The repo contains experimental paths for:

- prompt evolution,
- hypothesis generation,
- evidence validation through search,
- Obsidian bidirectional sync,
- confidence-gated code suggestions,
- isolated worktree self-improvement cycles.

These are not proven autonomous improvement. They are prototype control loops that require human review. Treat generated code, commits, graph edits, and confidence updates as inspectable suggestions, not trusted autonomous behavior.

## Known Limitations

- Extraction can be wrong, incomplete, or overly literal.
- Confidence scores are heuristic and not calibrated probabilities.
- Contradiction detection is incomplete and heuristic.
- LLM-backed extraction depends on local models where used.
- External ingestion depends on search availability and conservative rate limits.
- Large graphs may require the Docker-backed Memgraph path; the default NetworkX cache is intended for smaller local runs.
- Obsidian vault output can be noisy.
- Self-improvement workflows can modify code and commit when enabled; use dry-run paths and safe branches deliberately.

More detail: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Relationship To The TS Stack

- [TS-Start-Here](https://github.com/BoggersTheFish/TS-Start-Here): public ecosystem map and repo taxonomy.
- [TS-Reasoner-v0](https://github.com/BoggersTheFish/TS-Reasoner-v0): verifier-backed toy reasoning traces and repair telemetry.
- [TS-Codex-OS](https://github.com/BoggersTheFish/TS-Codex-OS): project graph, tension ledger, planner, and release receipts.
- [TS-Core](https://github.com/BoggersTheFish/TS-Core): graph/tension runtime kernel.

BoggersTheCIG is the claim/evidence/provenance branch. It stores and inspects knowledge state; it does not replace verifier-backed reasoning traces.

## Roadmap

Near-term:

- Make first-contact docs and limitations explicit.
- Stabilize minimal ingestion -> graph -> provenance -> contradiction examples.
- Add small reproducible CIG receipts.
- Clarify relationship between `BoggersTheCIG` and `cig-ts-engine`.

Later:

- Improve extraction evaluation.
- Add contradiction benchmark fixtures.
- Add cleaner bridge-detection receipts.
- Connect CIG state to TS-Reasoner trace/provenance examples.
