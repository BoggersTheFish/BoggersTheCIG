# BoggersTheCIG — TS Cognitive Engine

**A self-evolving, local-first knowledge graph engine** built on the **Thinking System (TS)** architecture.

It ingests information, builds a persistent epistemic graph, validates hypotheses with evidence, detects contradictions, and continuously self-improves — both its knowledge and its own code.

Everything runs locally. Zero cloud cost. Full transparency.

---

## Quick Start

```bash
git clone https://github.com/BoggersTheFish/BoggersTheCIG.git
cd BoggersTheCIG

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Start Ollama and pull a model
ollama pull qwen2.5-coder:7b

# Run a self-improvement cycle
python src/main.py --self-improve
```

See [Setup](#setup) for full details.

---

## Key Features

- **Epistemic Knowledge Graph** with confidence scoring, provenance, and Ebbinghaus decay
- **Evidence-based Hypothesis Validation** (web search + semantic similarity)
- **Bidirectional Obsidian Sync** — your manual edits strengthen the graph
- **Self-Code Modification** with isolated testing and coherence gates
- **Continuous Background Evolution** via GitHub Actions (or locally)
- **Rich API** + live dashboard
- **Coherence Guardrails** — automatic rollback on quality regression

---

## Architecture Overview

The system is built as a **13-stage pipeline** running in continuous loops:

1. **Language Layer** — Multi-prompt triple extraction with quality tracking
2. **Concept Graph** — SQLite primary store + NetworkX cache
3. **Provenance & Memory** — Full audit trail and decay mechanics
4. **Core Reasoning** — Causal chains, bridge nodes, contradiction detection
5. **Hypothesis Engine** — Gap detection → evidence validation
6. **Self-Improver** — Confidence-gated commits + code evolution
7. **Obsidian Integration** — Bidirectional human-AI collaboration
8. **API & Dashboard** — FastAPI endpoints and live metrics

Full details in the [Architecture section](#architecture) below.

---

## Project Status

**Active development** — This is the central engine of the entire TS ecosystem.

Most recent major upgrades (v2):
- Full epistemic reasoning system
- SQLite + provenance tracking
- Evidence-based validation
- Safer self-modification pipeline

---

## Setup

### Prerequisites
- Python 3.12+
- Ollama running
- 8GB+ RAM recommended

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Hardware-Aware Model Selection

The system auto-detects your hardware and recommends the best Ollama model.

---

## Running

**Basic usage**
```bash
python src/main.py --input "Your text here"
python src/main.py --self-improve
python src/main.py --self-improve --ingest-external
```

**API Server**
```bash
uvicorn src.api.app:app --reload
```

See full command reference in the original detailed README (still present in the repo for now).

---

## Links in the TS Ecosystem

- **BoggersTheAI** — The full Thinking System Operating System
- **TS-Core** — Reusable graph dynamics kernel
- **bozo** — TensionLM experimental model
- **GOAT-TS** — Theoretical cognitive architecture foundation

---

## Philosophy

We are not building another LLM wrapper.

We are building **systems that think more like reality thinks** — through constraint satisfaction, wave propagation, tension resolution, and continuous emergence.

---

**Made with obsessive love for cognitive architectures.**

Questions? Open an issue or reach out.
