# PDR-003: Task Corpus Approach

**Status:** Accepted
**Date:** 2026-06-05

## Context

The experiment requires 30 unique text passages for summarization, each 300–800 words. These passages are the task content: identical across all three prompt variants, all repetitions, and both models.

Properties the corpus needs:
- Varied enough that passages don't share distinctive opening phrases (which could trigger cross-passage caching)
- Topically diverse enough that the choice of passage doesn't systematically favor one prompt style
- No copyright concerns if the experiment is published
- Fully committed to the repo so results are reproducible

## Options Considered

### Synthetic generation (committed to repo)

Write 30 synthetic passages at build time — business memos, technical explainers, meeting notes, product descriptions, etc. — and commit `corpus/tasks.json` to the repo.

Pros: zero copyright risk, fully reproducible, passage content is transparent, can be designed to avoid duplicate openings
Cons: synthetic text may be less "realistic" than real-world writing

### User-provided (placeholder corpus)

Build the runner with a placeholder corpus format; the user supplies the 30 passages before running.

Pros: user controls content quality
Cons: blocks implementation until passages exist; harder to publish reproducibly

### Public domain text (sourced)

Pull passages from Project Gutenberg, Wikipedia, or similar public-domain sources.

Pros: natural language variation
Cons: sourcing and license verification adds work, passages may have repetitive stylistic fingerprints (e.g., Wikipedia's tone), complicates reproducibility if source URLs change

## Decision

Synthetic generation, committed to `corpus/tasks.json`. The passages cover 10 distinct topic areas (3 passages each): business operations, software engineering, project management, HR and hiring, product design, customer success, data infrastructure, security policy, finance, and marketing. Each passage has a unique first sentence to minimize cross-passage cache collisions.

The synthetic origin is disclosed in the methodology statement.

## Consequences

- **Positive:** Fully reproducible, no external dependencies, publishable without copyright concern, total control over content to avoid caching traps
- **Negative:** Text is "AI-like" by nature; a reader might argue real-world passages would produce different results
- **Risks:** Passages that are too similar in structure could affect token counts consistently across passages (e.g., if all memos start with "To:"). Mitigation: review passage openings for variety before running.
