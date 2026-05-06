# BoggersTheCIG — TS Cognitive Engine

**A self-evolving, local-first knowledge graph engine** built on the **Thinking System (TS)** architecture.

It continuously ingests information, extracts high-confidence concept triples, maintains an epistemic knowledge graph, validates hypotheses against evidence, applies Ebbinghaus-style forgetting, and even modifies its own code when coherence improves.

Everything runs locally with Ollama. Zero cloud cost.

---

## Quick Start

```bash
git clone https://github.com/BoggersTheFish/BoggersTheCIG.git
cd BoggersTheCIG

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Pull recommended model (auto-detected by hardware)
ollama pull qwen2.5-coder:7b

# Run one self-improvement cycle
python src/main.py --self-improve
```

See full setup in [Setup](#setup) below.

---

## What Makes It Different

- **Epistemic Graph**: Every edge carries confidence, provenance, timestamps, and source type
- **Evidence-Based Validation**: Hypotheses checked against real web results (DuckDuckGo)
- **Bidirectional Obsidian Sync**: Your manual edits in Obsidian flow back into the graph as high-confidence edges
- **Self-Modification**: Safely evolves its own code in isolated worktrees + test + coherence gate
- **Continuous Loop**: Runs via GitHub Actions every 5 hours (or locally)

---

## Core Architecture

The system is a **13-stage pipeline**:

1. Language Layer → Triple extraction (prompt registry + quality tracking)
2. Concept Graph → SQLite + NetworkX
3. Provenance & Memory Layers
4. Core Engine (causal chains, bridge nodes, contradictions)
5. Hypothesis Generator
6. Evidence Validation Engine
7. Self-Improver (confidence-gated commits + code evolution)
8. Obsidian bidirectional sync
9. Visualization & Insights
10. FastAPI dashboard + endpoints

Full details in the [original architecture section](https://github.com/BoggersTheFish/BoggersTheCIG/blob/main/README.md#architecture) (preserved for deep reference).

---

## Key Features

- **Confidence System** with corroboration boosts
- **Ebbinghaus forgetting** + human-edit protection
- **Semantic + structural coherence** metrics
- **Self-code evolution** with safety gates
- **Live API** (`/ask`, `/trace`, `/bridges`, `/contradictions`, etc.)
- **Graph dashboard**

---

## Setup & Running

**Full detailed setup** is in the [original long README](https://github.com/BoggersTheFish/BoggersTheCIG) for now — we're cleaning this progressively.

**Quick commands**:

```bash
python src/main.py --self-improve                  # Normal cycle
python src/main.py --self-improve --safe-evolve   # With code evolution
python src/main.py --self-improve --ingest-external
uvicorn src.api.app:app --reload                   # Start dashboard
```

---

## Part of the TS Ecosystem

- **BoggersTheAI** — The full Thinking System Operating System (wave propagation runtime)
- **TS-Core** — Lightweight graph dynamics kernel (Python + optional Rust)
- **bozo / TensionLM** — Experimental LLMs using tension graphs instead of softmax
- **GOAT-TS** — Theoretical cognitive architecture foundation

All repos are nodes in the same living system.

---

## Philosophy

We're not scaling next-token prediction.

We're building **substrate-level cognition**: constraint graphs → activation waves → tension resolution → emergent stable structures.

Language is just the surface pressure release.

---

**License**: MIT

**Status**: Actively evolving (Wave in progress)
