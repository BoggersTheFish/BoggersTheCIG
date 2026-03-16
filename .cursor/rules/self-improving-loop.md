# Self-Improving Loop Rules (TS Evolution)

When editing code that participates in the self-improving loop (self_improver.py, ollama_integration.py, continuous_thinker.py):

## Evolution Cycle

1. **Pull** — Always pull latest from git before analysis.
2. **Analyze** — Use Ollama (local) to detect tensions, gaps, and improvement opportunities per ts-evolution principles.
3. **Generate** — Produce real code diffs, not placeholders. Every change must be functional.
4. **Validate** — Run pytest before committing. No commits with failing tests.
5. **Commit** — Atomic commits with clear messages. Include before/after metrics when applicable.
6. **Push** — Push only after local tests pass.

## Tension Detection (Ollama Prompts)

- Structural: cycles, contradictions, orphan nodes.
- Performance: slow paths, redundant embeddings, N+1 queries.
- Coverage: missing error handling, untested branches.
- Alignment: ethical drift, bias in generated content.

## Code Generation Constraints

- Never use stubs, TODOs, or placeholder comments.
- Every function must execute and return deterministic results.
- Respect graph-stability.mdc: no graph mutations without conflict checks.
- Respect ethical-safegaurds.mdc: all hypothesis/validation paths include alignment checks.
- Respect cost-aware-design.mdc: local-first, no paid APIs.

## Integration Points

- Obsidian vault at `obsidian/TS-Knowledge-Vault/` is the visual memory layer.
- Auto-export graph changes to Markdown with wikilinks after each loop iteration.
- Pre-commit hook runs TS coherence check (graph + eval) before allowing commits.
