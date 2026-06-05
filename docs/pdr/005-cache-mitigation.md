# PDR-005: Cache Mitigation Strategy

**Status:** Accepted
**Date:** 2026-06-05

## Context

Provider-side prompt caching can silently inflate or deflate token counts in ways that have nothing to do with prompt politeness. If a cached response is served for a polite prompt but not a bare prompt (or vice versa), the token difference reflects caching behavior, not politeness.

Both providers expose cache detection:
- **Anthropic:** `usage.cache_read_input_tokens` — non-zero when a cached prompt prefix was used
- **OpenAI:** `usage.prompt_tokens_details.cached_tokens` — non-zero when a cached prefix was used

## Approach

### 1. Randomize trial order

Generate all 900 trial definitions upfront. Shuffle them into a random order before executing. This prevents the same prompt from being sent in tight succession (which is the most common cause of cache hits). The shuffle seed is logged to `results/run_log.jsonl` for reproducibility.

### 2. Log `cached_tokens` per trial

Every trial record includes a `cached_tokens` field. The runner extracts this from `usage.cache_read_input_tokens` (Anthropic) or `usage.prompt_tokens_details.cached_tokens` (OpenAI), defaulting to 0 if the field is absent.

### 3. Dual result views

The analyzer produces two views of every table:
- **Raw:** All 900 trials included
- **Clean:** Trials where `cached_tokens > 0` are excluded

The article reports the clean view as the primary result and includes the raw view in an appendix.

### 4. What counts as a cache hit

A trial is flagged as cache-affected if `cached_tokens > 0` at the time of the API response. There is no attempt to pre-warm or pre-bust caches — the experiment runs in natural API conditions and detects caches after the fact.

## What this does NOT do

This strategy does not prevent cache hits — it detects and filters them. If a large fraction of trials are cache-affected (unlikely given randomization but possible), the clean dataset may be smaller than expected. The methodology statement will report the cache hit rate.

## Consequences

- **Positive:** Clean results are provably cache-free; raw results remain available for comparison
- **Negative:** Aggressive caching could reduce the clean dataset size; the experiment is designed with 5 reps specifically to absorb some attrition
- **Risks:** Provider caching behavior may change. Mitigation: cache detection is field-based and gracefully defaults to 0 if the field is absent.
