# Architecture

BoggersTheCIG is CIG infrastructure: a local-first provenance-aware concept graph for confidence-weighted claims, evidence, contradictions, and inspectable knowledge state.

## Pipeline

```text
text ingestion
  -> triple extraction
  -> concept graph
  -> provenance/confidence store
  -> contradiction detection
  -> bridge detection
  -> Obsidian sync
  -> API / visualization
```

## Text Ingestion

Text can enter from direct CLI input, local files, Obsidian sync, or optional external ingestion helpers.

The ingestion layer should be treated as lossy. Source text can produce incomplete or malformed triples, especially when local model extraction is enabled.

## Triple Extraction

The system extracts subject/relation/object triples. Extraction may be rule-based or backed by an optional local LLM through Ollama.

Triples should be inspected as candidate claims, not ground truth.

## Concept Graph

Extracted triples are stored as graph edges between concepts. Edges can carry:

- relation type,
- confidence,
- source type,
- provenance,
- timestamps,
- reinforcement metadata.

SQLite is used as a persistent store; NetworkX is used as a local graph/cache path. Larger graph paths may use Docker-backed graph services where configured.

## Provenance / Confidence Store

The provenance layer records where a claim came from and how it has been reinforced. Confidence is heuristic and should not be read as a calibrated probability.

## Contradiction Detection

Contradiction detection currently uses heuristic structural and semantic signals. It can surface useful conflicts, but it is incomplete and can miss or over-detect contradictions.

## Bridge Detection

Bridge detection identifies nodes or edges that connect otherwise separate graph regions. These are useful inspection targets, not proof of causal importance.

## Obsidian Sync

The Obsidian layer exports graph state into markdown notes so humans can inspect concepts, relations, confidence, provenance, and bridge tags.

The vault can become noisy because it reflects extracted graph state rather than a curated knowledge base.

## API / Visualization Layer

The FastAPI layer exposes graph inspection endpoints such as query, trace, contradictions, bridges, and metrics where available.

Visualization paths are intended for inspection and debugging, not for proving correctness.

## Optional Ollama / Local LLM Extraction

Ollama-backed extraction can improve coverage but introduces model errors and local environment dependencies. It requires an installed model and local server.

The repo should remain usable in narrower rule-based paths where possible.

## Experimental Self-Improvement Loop

The self-improvement path can generate suggestions, use tests or coherence checks, and stage or commit changes when enabled.

This path is experimental. It is not proven autonomous improvement and should not be treated as a safe self-running agent. Human review is required.
