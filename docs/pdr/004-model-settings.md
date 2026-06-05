# PDR-004: Model Settings for API Calls

**Status:** Accepted
**Date:** 2026-06-05

## Context

The experiment's primary question is whether polite prompt framing changes how much a model *thinks* — measured via reasoning tokens — not merely how much it writes. This changes the model settings significantly from a "suppress variance" approach to a "let the model reason naturally" approach.

Reasoning tokens are a primary metric alongside input and output tokens.

## Decision

### Claude: `claude-opus-4-8`

```python
client.messages.create(
    model="claude-opus-4-8",
    thinking={"type": "adaptive"},
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}]
)
```

- **Adaptive thinking**: Claude decides how much to think based on the prompt framing. This is the signal being measured.
- **No temperature parameter**: Opus 4.8 does not accept `temperature` — it returns a 400 error. This is acceptable; the experiment is measuring natural variance in reasoning, not trying to suppress it.
- **`max_tokens=4096`**: High ceiling so thinking + output are never artificially truncated. Thinking tokens are tracked separately from output tokens on Claude and do not count toward this limit.

### GPT-5.5: `gpt-5.5`

```python
client.chat.completions.create(
    model="gpt-5.5",
    max_completion_tokens=4096,
    messages=[{"role": "user", "content": prompt}]
)
```

- **No `reasoning_effort` override**: Default behavior (medium effort) lets GPT-5.5 reason as much as the prompt naturally elicits — the same philosophy as Claude's adaptive thinking.
- **No `temperature` override**: Default temperature on GPT-5.5 is used. Setting temperature=0 would suppress reasoning variance, which is the opposite of what the experiment is measuring.
- **`max_completion_tokens=4096`**: On GPT-5.5, reasoning tokens count toward this limit (unlike Claude where they are separate). High ceiling prevents truncation.

## Key Asymmetry: Token Accounting

| | Claude Opus 4.8 | GPT-5.5 |
|---|---|---|
| Thinking/reasoning tokens | Separate from `max_tokens` | Count toward `max_completion_tokens` |
| Billed as | Output tokens (same rate) | Output tokens (same rate) |
| Exposed in API response | `usage.thinking_tokens` | `usage.completion_tokens_details.reasoning_tokens` |

The JSONL record captures both as `reasoning_tokens`. The analyzer treats them equivalently for comparison purposes — both are billed as output tokens and both reflect internal reasoning effort.

## Why not temperature=0?

Temperature=0 suppresses output variance. This experiment measures whether prompt politeness changes reasoning behavior — which is itself a form of variance. Flattening that variance with temperature=0 would hide the signal being measured. The 5 repetitions per variant exist to capture the distribution of reasoning token usage across runs, not to average away noise.

## Why Opus 4.8 instead of Opus 4.6?

The original PDR-004 chose Opus 4.6 for `temperature=0` support. Now that thinking is the primary metric and temperature suppression is undesirable, there is no reason to use the prior-generation model. Opus 4.8 is the current flagship with native adaptive thinking support.

## Consequences

- **Positive:** Both models reason as naturally as the prompt elicits; reasoning tokens are a meaningful, unmanipulated signal
- **Negative:** Cost is variable and harder to predict — a polite prompt that triggers 2000 reasoning tokens costs ~10× more than a bare prompt that triggers 200. Pre-run cost estimation is approximate.
- **Risks:** Adaptive thinking on Claude 4.8 means some trials may produce very long thinking blocks, which would dwarf the output token difference from politeness. Mitigation: the analyzer separates input, output, and reasoning tokens so the article can report each independently.
