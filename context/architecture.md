# Architecture: Politeness Experiment

**Last updated:** 2026-06-05
**Status:** Accepted

## Overview

A single-purpose research tool that runs 900 API calls across two frontier models to measure whether polite prompt phrasing changes token usage and cost. 30 summarization tasks × 3 prompt variants × 5 repetitions × 2 models. Results are logged per-trial and analyzed in two views: raw (all runs) and clean (cache-hit runs excluded).

## Architectural Style

Single-package Python script with modular provider wrappers. No framework overhead — this runs once, produces a dataset, and exits.

## Components

| Component | Responsibility |
|-----------|---------------|
| `corpus/tasks.json` | 30 synthetic text passages + metadata used as task content |
| `src/runner.py` | Orchestrates all 900 trials; handles resumption, randomization, progress |
| `src/models/anthropic.py` | Claude Opus 4.8 API wrapper; returns normalized usage dict including thinking tokens |
| `src/models/openai.py` | GPT-5.5 API wrapper; returns normalized usage dict |
| `src/analyze.py` | Loads raw JSONL; produces per-variant statistics; writes CSV reports |
| `results/raw.jsonl` | Append-only trial log; one JSON object per line |
| `results/analysis/` | Post-hoc CSV and text reports |

## Data Architecture

| Store | Type | Owner | Why |
|-------|------|-------|-----|
| `results/raw.jsonl` | JSONL file | runner | Append-only; safe for partial runs; trivially loadable with pandas |
| `results/analysis/*.csv` | CSV files | analyzer | Human-readable summary tables for the article |
| `corpus/tasks.json` | JSON file | static | Immutable during a run; sourced once at build time |

There is no database. At 900 records with ~10 fields each, relational overhead is pure cost.

**Trial identity:** Each trial is uniquely identified by the composite key `(model, task_id, variant, rep)`. The runner hashes this to a `trial_id` string and uses it to skip already-completed trials on resume.

**Cache detection:** Each trial record includes `cached_tokens` (from `usage.cache_read_input_tokens` on Anthropic; from `usage.prompt_tokens_details.cached_tokens` on OpenAI). The analyzer produces two views: raw (all 900 trials) and clean (trials where `cached_tokens == 0`).

## JSONL Record Schema

```json
{
  "trial_id": "claude-opus-4-6|T01|bare|1",
  "model": "claude-opus-4-6",
  "task_id": "T01",
  "variant": "bare",
  "rep": 1,
  "prompt_text": "Summarize the following text:\n\n...",
  "response_text": "...",
  "input_tokens": 156,
  "output_tokens": 89,
  "reasoning_tokens": 0,
  "cached_tokens": 0,
  "total_tokens": 245,
  "cost_usd": 0.00347,
  "timestamp": "2026-06-05T14:23:01Z",
  "latency_ms": 1234,
  "quality": {
    "completed_task": true,
    "usable_summary": true,
    "unnecessary_fluff": false,
    "formatting_issues": false
  }
}
```

`reasoning_tokens` is a **primary metric** — it measures how much internal thinking the model did. On Claude it comes from `usage.thinking_tokens`; on GPT-5.5 from `usage.completion_tokens_details.reasoning_tokens`. Both are billed at the output token rate.

Quality fields are populated by a post-hoc pass (not inline during the run — don't block 900 calls on manual review). Leave them null on initial write; fill them via a separate `review` command.

## Key Decisions

- [PDR-001: Storage Format](../docs/pdr/001-storage-format.md) — JSONL over SQLite or CSV; append-only, resumable, zero setup
- [PDR-002: Execution Model](../docs/pdr/002-execution-model.md) — Sequential with optional concurrency; sequential is the safe default for a one-shot experiment
- [PDR-003: Task Corpus](../docs/pdr/003-task-corpus.md) — Synthetic passages generated at build time; committed to repo as static corpus
- [PDR-004: Model Settings](../docs/pdr/004-model-settings.md) — `claude-opus-4-8` with adaptive thinking; `gpt-5.5` with default reasoning effort; reasoning tokens are the primary metric
- [PDR-005: Cache Mitigation](../docs/pdr/005-cache-mitigation.md) — Randomized trial order + per-trial cache detection + clean/raw dual view

## Failure Model

| Failure | Behavior |
|---------|----------|
| API error mid-run | Trial is not written to JSONL; runner logs the error and continues to next trial |
| Rate limit | Runner backs off with exponential retry (max 3 attempts); logs if all retries fail |
| Process killed mid-run | Resume by re-running; completed trials are skipped via trial_id deduplication |
| Partial JSONL line | Last line may be truncated if process dies mid-write; runner validates each line on load and skips malformed ones |

## Cost Estimate

| Model | Calls | Est. Input | Est. Output+Thinking | Est. Total |
|-------|-------|-----------|---------------------|-----------|
| claude-opus-4-8 | 450 | ~270K tokens | ~68K–500K tokens | $1.35–$8.00 |
| gpt-5.5 | 450 | ~270K tokens | ~68K–500K tokens | $1.35–$9.00 |
| **Total** | **900** | | | **~$10–$30** |

Cost is variable because reasoning tokens are the signal being measured — polite prompts may trigger significantly more thinking than bare prompts. The range above spans minimal reasoning (summary only, no thinking) to moderate reasoning (a few hundred thinking tokens per call). Set an API spend limit before running.

## Non-Functional Properties

| Property | Target | Notes |
|----------|--------|-------|
| Runtime | ~30 min sequential | ~6 min at concurrency=5 |
| Resumability | Full | Any interrupted run resumes from last completed trial |
| Reproducibility | High | Randomization seed logged per run; temperature=0 on both models |
| Cost | ~$6–10 | See estimate above; set API spend limits before running |

## Integrations

| System | How | Auth |
|--------|-----|------|
| Anthropic API | `anthropic` Python SDK | `ANTHROPIC_API_KEY` env var |
| OpenAI API | `openai` Python SDK | `OPENAI_API_KEY` env var |

## Operational Notes

- Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` before running
- Run `python src/runner.py` to execute all 900 trials
- Run `python src/analyze.py` after completion to produce reports
- Use `--model claude-opus-4-6` or `--model gpt-5.5` to run one model at a time
- Use `--dry-run` to validate corpus and print trial count without making API calls
- Results accumulate in `results/raw.jsonl`; never delete this file mid-experiment

## Open Questions

- Quality check fields (`completed_task`, `usable_summary`, etc.) — manual review or automated? The design spec lists these as fields but doesn't specify how they're populated. Current design leaves them null until a separate review pass.
- Should the randomization seed be fixed (for full reproducibility) or random-per-run (for better cache mitigation)? Current design: random-per-run, with seed logged to `results/run_log.jsonl`.
