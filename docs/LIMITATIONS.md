# Limitations

BoggersTheCIG is an active experimental prototype, not a production knowledge base or autonomous reasoner.

## Extraction Errors

Text-to-triple extraction can be wrong, incomplete, overly literal, or inconsistent across runs. Local LLM extraction can hallucinate structure that is not present in the source.

## Heuristic Confidence Scores

Confidence scores are heuristic signals. They are not calibrated truth probabilities and should not be used as authoritative factual scores.

## Incomplete Contradiction Detection

Contradiction detection is incomplete. The system can miss contradictions, over-detect conflicts, or confuse relation direction and meaning.

## Local Model Dependencies

LLM-backed paths depend on local Ollama models where used. Results can vary by model, hardware, prompt, and environment.

## No General Reasoning Claim

This repo does not claim general reasoning ability. It stores and inspects claim/evidence graph state.

## No Production KB Claim

The graph is not a production knowledge base. It is a local research artifact for inspecting provenance, confidence, contradictions, and memory shape.

## Experimental Self-Improvement

Self-improvement paths are experimental. They are not proven autonomous improvement and are not safe to treat as a self-running agent. Generated changes require human review.

## Obsidian Vault Noise

The Obsidian vault may be noisy because it reflects extracted graph state. It should be treated as an inspection surface, not a curated encyclopedia.

## External Ingestion Limits

External ingestion depends on search availability, rate limits, and source quality. Missing or low-quality source data can distort the graph.
