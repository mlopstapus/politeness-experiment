# Article Design: Politeness Experiment

**Date:** 2026-06-08
**Status:** Approved
**Author:** Ben Anderson
**Publication targets:** anchorstack.dev (canonical), X article, LinkedIn article

---

## Overview

A three-piece content package reporting the results of a controlled experiment testing whether prompt politeness affects LLM reasoning effort, accuracy, and cost. The experiment ran 452 trials across Claude Opus 4.8, Claude Sonnet 4.6, and GPT-5.5 using three prompt variants (bare / polite / overly polite) on 10 math and logic reasoning tasks.

The politeness finding is the hook; the model economics story is the payoff.

---

## Key Data Points (use these verbatim)

| Model | Avg reasoning tokens | Accuracy | Cost/trial | Cost/correct answer |
|---|---|---|---|---|
| Claude Opus 4.8 | 1,004 | 100% | $0.047 | $0.047 |
| Claude Sonnet 4.6 | 3,251 | 94% | $0.060 | $0.064 |
| GPT-5.5 | 659 | 97.3% | $0.036 | $0.037 |

**Politeness effect on Opus (cost, same 100% accuracy):**
- Bare: $0.050/trial
- Polite: $0.046/trial (saves 6%)
- Overly polite: $0.045/trial (saves 10%)

**Task difficulty range (Opus, reasoning tokens):**
- R01 (train/bird problem): ~55 tokens
- R06 (knights and knaves): ~3,700 tokens
- Ratio: ~67x difference driven entirely by task content, not prompt framing

**Total experiment cost:** $21.56

---

## Three Story Lenses

The full article is structured around three cross-sections. Cost and accuracy are the two metrics that run through all three.

### Lens 1 — Politeness
**Question:** Does how you ask change what you get?
**Finding:** Accuracy is completely flat across all three variants for every model. But on Opus, politeness reduces reasoning effort — and therefore cost — by up to 10% with zero quality loss.
**Punchline:** Politeness doesn't make AI smarter. It makes Opus slightly cheaper to think.

### Lens 2 — Question Type
**Question:** What actually drives reasoning effort?
**Finding:** Task difficulty dwarfs prompt framing. The same model uses 67x more reasoning tokens on a hard logic puzzle than an easy arithmetic problem, regardless of how politely you asked. Politeness is noise next to this signal.
**Punchline:** The biggest lever on your AI bill isn't how you phrase your prompt — it's what you're asking it to do.

### Lens 3 — Model Type
**Question:** Which model gives the best value on reasoning tasks?
**Finding:** Sonnet uses 3x the reasoning tokens of Opus, costs more per trial, and gets more wrong — making it the most expensive option despite its reputation as the "budget" model. GPT-5.5 uses the fewest tokens, finishes second in accuracy, and is the cheapest per correct answer.
**Punchline:** The model you think is saving you money probably isn't.

---

## Piece 1: anchorstack.dev (canonical, full methodology)

**Target length:** ~1,500 words
**Tone:** Informed and accessible. Reads like a smart practitioner explaining what they found, not a paper abstract.
**Figures to include:** cost vs accuracy scatter, per-task reasoning heatmap, Opus per-task politeness effect

### Structure

1. **Cold open** (~100 words): The "should you say please to AI" debate. People genuinely argue about this. Frame as: someone decided to actually test it.
2. **The experiment** (~150 words): What was tested — three models, three prompt variants, 10 reasoning tasks, 5 reps each. Explain reasoning tokens with one plain-English analogy (the scratchpad the model uses before it answers). Total: 452 trials, $21.56.
3. **Lens 1 — Politeness** (~250 words): Accuracy is flat. But Opus costs less when you're polite. Introduce the "it pays to be nice" frame here.
4. **Lens 2 — Question type** (~300 words): The bigger surprise. 55 tokens vs 3,700 tokens, same model, same politeness. What you ask matters enormously more than how you ask it. Include the per-task reasoning heatmap.
5. **Lens 3 — Model type** (~400 words): The Sonnet bombshell. Walk through the cost-per-correct-answer table. GPT-5.5 as the quiet winner. Note the Sonnet reasoning inflation paradox (more thinking, worse answers, higher cost).
6. **Kicker** (~100 words): Bring back the politeness frame. Being nice to Opus is worth a few cents. Knowing which model to use is worth a lot more. Link to repo.

---

## Piece 2: X Article

**Target length:** ~600 words
**Tone:** Casual, punchy, conversational. Written like a good tweet thread that got promoted to an article.
**Lead:** Politeness hook — the experiment premise and the "it pays to be nice" finding.
**Methodology:** One sentence only. No figures — key numbers as inline callouts.
**What to omit:** Question type lens (save it as a "read the full piece" incentive). Detailed methodology.
**Ends with:** Link to anchorstack.dev for full methodology and data.

### Structure

1. Hook: The polite prompting debate — we tested it
2. The setup in one sentence
3. Finding: Accuracy doesn't change
4. Finding: But Opus costs ~10% less when you're polite — "it literally pays to be nice"
5. The bigger surprise: Sonnet is the most expensive model per correct answer ($0.064 vs $0.047 for Opus)
6. GPT-5.5 as the quiet value winner
7. CTA to full piece

---

## Piece 3: LinkedIn Article

**Target length:** ~700 words
**Tone:** Professional but not dry. Aimed at practitioners making model selection decisions.
**Lead:** Model economics — Sonnet bombshell first.
**Methodology:** One short paragraph for credibility.
**What to emphasize:** Cost per correct answer table, practical model selection implications.
**Ends with:** Link to anchorstack.dev for full methodology and data.

### Structure

1. Hook: "We found that Claude Sonnet — the 'cheaper' model — costs 37% more per correct answer than Opus"
2. Brief methodology paragraph
3. The Sonnet finding in detail (reasoning inflation + accuracy drop = cost blowout)
4. GPT-5.5 as the practical alternative
5. The politeness finding as a secondary takeaway (Opus cost lever)
6. Question type as context: what actually drives your AI bill
7. CTA to full piece

---

## Voice and Style Notes

- **Avoid:** "fascinating," "delve," "it's worth noting," "in conclusion"
- **Use:** Short sentences for key findings. Numbers as callouts, not buried in prose.
- **Methodology tone:** Confident but not defensive. The experiment is well-designed; explain it matter-of-factly.
- **The Sonnet finding:** Present as a discovery, not a criticism of Anthropic. The story is about how reasoning token inflation changes the economics — not that Sonnet is bad.
- **Analogies to use:** Reasoning tokens as "the scratchpad work before the answer." Task difficulty as "you wouldn't expect a calculator to sweat equally on 2+2 and a differential equation."

---

## Assets

| Asset | Used in |
|---|---|
| Cost vs accuracy figure (`07_cost_vs_accuracy.png`) | anchorstack.dev |
| Per-task reasoning heatmap (`04_task_difficulty_heatmap.png`) | anchorstack.dev |
| Opus per-task politeness effect (`05_opus_per_task_politeness_effect.png`) | anchorstack.dev |
| Cost-per-correct-answer table (generate from data) | All three pieces |

---

## Out of Scope

- Statistical significance testing (the within-variant variance is large; acknowledge this honestly rather than overclaiming)
- Generalizing beyond reasoning/math tasks — the corpus is 10 tasks, results may differ for creative or open-ended prompts
- Prompt engineering advice beyond the three findings
