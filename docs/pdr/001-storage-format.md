# PDR-001: Storage Format for Trial Results

**Status:** Accepted
**Date:** 2026-06-05

## Context

The experiment produces 900 trial records. Each record includes the full prompt text, response text, token counts, cost, and quality flags. The runner must be able to resume from any point without re-running completed trials. The analyzer needs to load all records and compute per-variant statistics.

## Options Considered

### JSONL (newline-delimited JSON)

Each trial is one JSON object on its own line. The runner appends a line per completed trial. The analyzer reads all lines with `pandas.read_json(..., lines=True)`.

Pros: zero setup, append-only so crash-safe, human-readable, trivially loadable, supports arbitrary nested fields (e.g., `quality` subobject)
Cons: no query language; full-file scan for deduplication check on resume (900 lines is negligible)

### SQLite

A local relational database with a `trials` table.

Pros: SQL queries, indexed lookups, transactional writes
Cons: requires schema definition, migration if fields change, additional dependency, no benefit at 900 rows, harder to inspect manually

### CSV

Flat file with one row per trial.

Pros: Excel-compatible
Cons: append operations can corrupt headers, no support for nested fields (quality subobject would need flattening), fragile if process dies mid-write

## Decision

JSONL. The dataset is 900 records, the fields are nested (quality subobject), and resumability requires only a linear scan of completed trial IDs at startup. SQLite's benefits don't materialize until the dataset is large enough to need indexed queries — which this never will be.

## Consequences

- **Positive:** No setup, no schema, safe appends, works with standard Python tooling (`json`, `pandas`)
- **Negative:** Resume scan reads all completed records on startup (fast at 900 records; irrelevant concern)
- **Risks:** Truncated last line if process is killed mid-write. Mitigation: runner validates each line on load and skips malformed entries; the trial will simply re-run on the next execution.
