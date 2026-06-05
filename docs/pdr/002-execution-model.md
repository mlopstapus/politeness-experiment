# PDR-002: Execution Model

**Status:** Accepted
**Date:** 2026-06-05

## Context

The runner must make 900 API calls. The key questions are: sequential or concurrent, and how to handle partial failures.

Expected sequential runtime: ~900 calls × ~2s average latency = ~30 minutes. At concurrency=5: ~6 minutes.

Both models have generous rate limits (GPT-5.5 Tier 1: 500 RPM; Claude Opus 4.6: similar). Neither model is a bottleneck at sane concurrency levels.

## Options Considered

### Sequential

Run one trial at a time. Next trial starts when the previous completes.

Pros: trivially simple, easy to reason about failure, no partial-write race conditions, predictable behavior for a one-shot experiment
Cons: ~30 minutes wall-clock time

### Async concurrent (asyncio)

Run N trials in parallel using `asyncio` + `httpx` or async SDK clients.

Pros: 5x–10x faster; at concurrency=5 this is ~6 minutes
Cons: more complex error handling, slightly more complex resumption logic, requires async SDK usage

### Threaded concurrent

Use `concurrent.futures.ThreadPoolExecutor`.

Pros: works with sync SDK clients without async rewrite
Cons: threading overhead, GIL applies to CPU-bound parts (negligible here since it's I/O-bound), harder to reason about than asyncio

## Decision

Sequential by default, with `--concurrency N` flag to enable async mode. The experiment runs once (or a few times). The 30-minute runtime is acceptable for a research run. Sequential is the easiest to debug if something goes wrong, and the most trustworthy for ensuring clean independent measurements.

The `--concurrency` flag exists for operators who want the faster path and are comfortable with async behavior.

## Consequences

- **Positive:** Simple implementation, predictable behavior, easy to trace failures in output
- **Negative:** ~30 minutes for a full run in default mode
- **Risks:** A model API going down mid-run stalls the sequential runner. Mitigation: exponential retry (max 3 attempts per trial), then log-and-skip. The trial ID is not written to JSONL on failure, so it will re-run on resume.
