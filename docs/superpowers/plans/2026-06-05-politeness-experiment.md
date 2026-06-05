# Politeness Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research tool that makes 1,350 API calls across three LLM models to measure whether polite prompt framing changes reasoning/thinking token usage.

**Architecture:** Python CLI with a shared schema, three model wrappers (Claude Opus 4.8, Claude Sonnet 4.6, GPT-5.5), a sequential resumable runner that appends to JSONL, and a pandas-based analyzer that exports long-format CSVs for visualization. Reasoning tokens are the primary metric — they measure how much internal thinking each prompt variant triggers.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `openai` SDK, `pandas`, `tqdm`, `pytest`, `pytest-mock`, `python-dotenv`

---

## File Structure

```
politeness-experiment/
├── corpus/
│   └── tasks.json                    # 30 synthetic text passages (static)
├── src/
│   ├── __init__.py
│   ├── schema.py                     # TrialRecord TypedDict, constants, make_trial_id
│   ├── prompts.py                    # Prompt variant builder
│   ├── costs.py                      # Token cost calculator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anthropic.py              # Claude Opus 4.8 + Sonnet 4.6 wrapper
│   │   └── openai.py                 # GPT-5.5 wrapper
│   ├── runner.py                     # Trial orchestrator (resumable, randomized)
│   └── analyze.py                    # JSONL → long-format CSV reports
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── sample_results.jsonl      # 9-line fixture for analyzer tests
│   ├── test_schema.py
│   ├── test_prompts.py
│   ├── test_costs.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_anthropic.py
│   │   └── test_openai.py
│   ├── test_runner.py
│   └── test_analyze.py
├── results/
│   ├── .gitkeep
│   └── analysis/
│       └── .gitkeep
├── docs/
│   ├── pdr/
│   └── superpowers/plans/
├── context/
│   └── architecture.md
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "politeness-experiment"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.env.example`**

```
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here
```

- [ ] **Step 3: Create init files**

```bash
touch src/__init__.py src/models/__init__.py
touch tests/__init__.py tests/test_models/__init__.py
touch tests/fixtures/.gitkeep
```

- [ ] **Step 4: Replace `requirements.txt`**

```
anthropic>=0.52.0
openai>=1.82.0
pandas>=2.2.0
tqdm>=4.67.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-mock>=3.14.0
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Verify pytest discovers test paths**

```bash
cd ~/repos/politeness-experiment && pytest --collect-only
```

Expected: "no tests ran" — no test files yet, that's correct.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example requirements.txt src/__init__.py src/models/__init__.py tests/__init__.py tests/test_models/__init__.py tests/fixtures/.gitkeep
git commit -m "chore: project scaffold — pytest, deps, directory structure"
```

---

### Task 2: Schema

**Files:**
- Create: `src/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_schema.py`:

```python
import json
from src.schema import TrialRecord, MODELS, VARIANTS, REPS, make_trial_id


def test_models_list():
    assert "claude-opus-4-8" in MODELS
    assert "claude-sonnet-4-6" in MODELS
    assert "gpt-5.5" in MODELS
    assert len(MODELS) == 3


def test_variants_list():
    assert VARIANTS == ["bare", "polite", "overly_polite"]


def test_reps():
    assert REPS == 5


def test_trial_id_format():
    tid = make_trial_id("claude-opus-4-8", "T01", "bare", 1)
    assert tid == "claude-opus-4-8|T01|bare|1"


def test_trial_record_is_json_serializable():
    record: TrialRecord = {
        "trial_id": make_trial_id("gpt-5.5", "T01", "polite", 2),
        "model": "gpt-5.5",
        "task_id": "T01",
        "variant": "polite",
        "rep": 2,
        "prompt_text": "Please summarize the following text:\n\nhello world",
        "response_text": "A greeting.",
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 15,
        "cost_usd": 0.000125,
        "timestamp": "2026-06-05T12:00:00Z",
        "latency_ms": 350,
        "quality": None,
    }
    deserialized = json.loads(json.dumps(record))
    assert deserialized["trial_id"] == "gpt-5.5|T01|polite|2"
    assert deserialized["reasoning_tokens"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_schema.py -v
```

Expected: `ImportError` — `src.schema` does not exist yet.

- [ ] **Step 3: Create `src/schema.py`**

```python
from typing import TypedDict, Optional

MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5"]
VARIANTS = ["bare", "polite", "overly_polite"]
REPS = 5


def make_trial_id(model: str, task_id: str, variant: str, rep: int) -> str:
    return f"{model}|{task_id}|{variant}|{rep}"


class TrialRecord(TypedDict):
    trial_id: str
    model: str
    task_id: str
    variant: str
    rep: int
    prompt_text: str
    response_text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str
    latency_ms: int
    quality: Optional[dict]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_schema.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/schema.py tests/test_schema.py
git commit -m "feat: schema — TrialRecord TypedDict, constants, make_trial_id"
```

---

### Task 3: Task Corpus

**Files:**
- Create: `corpus/tasks.json`
- Create: `tests/test_corpus.py`

- [ ] **Step 1: Write failing corpus tests**

Create `tests/test_corpus.py`:

```python
import json
from pathlib import Path

CORPUS_PATH = Path("corpus/tasks.json")


def load_corpus():
    with CORPUS_PATH.open() as f:
        return json.load(f)


def test_corpus_exists():
    assert CORPUS_PATH.exists()


def test_corpus_has_30_tasks():
    assert len(load_corpus()) == 30


def test_each_task_has_required_fields():
    for task in load_corpus():
        assert "task_id" in task
        assert "title" in task
        assert "content" in task
        assert "topic" in task


def test_task_ids_are_unique_and_sequential():
    corpus = load_corpus()
    ids = sorted(t["task_id"] for t in corpus)
    assert ids == [f"T{i:02d}" for i in range(1, 31)]


def test_content_word_count_in_range():
    for task in load_corpus():
        wc = len(task["content"].split())
        assert 200 <= wc <= 900, f"{task['task_id']} has {wc} words"


def test_no_duplicate_first_sentences():
    corpus = load_corpus()
    first = [t["content"].split(".")[0] for t in corpus]
    assert len(first) == len(set(first)), "Duplicate first sentences — cache collision risk"


def test_ten_topic_areas():
    assert len(set(t["topic"] for t in load_corpus())) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_corpus.py -v
```

Expected: `test_corpus_exists` FAILS.

- [ ] **Step 3: Create `corpus/tasks.json`**

Generate 30 passages across 10 topic areas (3 per topic), each 300–800 words with a unique opening sentence. Topic areas and task IDs:

| Topic | task_ids |
|-------|---------|
| `cloud_infrastructure` | T01–T03 |
| `software_engineering` | T04–T06 |
| `project_management` | T07–T09 |
| `hr_hiring` | T10–T12 |
| `product_design` | T13–T15 |
| `data_infrastructure` | T16–T18 |
| `security_policy` | T19–T21 |
| `customer_success` | T22–T24 |
| `finance_budget` | T25–T27 |
| `marketing` | T28–T30 |

Schema for each entry:
```json
{
  "task_id": "T01",
  "title": "Migrating a Monolith to Microservices",
  "topic": "cloud_infrastructure",
  "content": "..."
}
```

Two complete example passages to anchor style and length:

**T01 (~420 words, cloud_infrastructure):**
```
Decomposing a production monolith into microservices is one of the most disruptive infrastructure decisions an engineering team can make, and the failure rate is high when teams underestimate the organizational change required.\n\nThe technical work is only half the problem. The other half is agreeing on service boundaries before writing a single line of infrastructure code. Service boundaries that map to team ownership tend to survive. Boundaries drawn by technical layer — separating all database logic from all business logic, for instance — almost always collapse under the weight of cross-cutting concerns.\n\nThe strangler fig pattern offers the lowest-risk path. Rather than rewriting the monolith from scratch, new functionality is built as standalone services while the monolith continues to handle existing traffic. A routing layer gradually shifts requests away from the monolith as each service reaches production readiness. The monolith shrinks over time rather than being replaced in a single cutover event.\n\nData is the hardest part of any decomposition. Each microservice ideally owns its data store, but in practice, many monoliths have years of tangled joins and shared tables that cannot be cleanly divided. Teams that attempt to split a shared database before the services are stable invariably break something in production. The safer sequence is to extract the service first, give it its own data access layer pointing at the shared database, and only then migrate the underlying data to a dedicated store.\n\nOperational readiness is frequently underestimated. A monolith requires one deployment pipeline, one log aggregation setup, and one set of runbooks. A ten-service system requires ten of each. Distributed tracing, service mesh configuration, and per-service alerting add up quickly. Teams should budget at least as much time for observability as for the service extraction itself.\n\nFinally, migration timelines almost always slip. The work is inherently exploratory — teams discover unexpected dependencies only after they start pulling threads. Building in explicit discovery phases, where the goal is to map what exists rather than to ship, helps calibrate the schedule before the pressure to deliver takes over.
```

**T16 (~350 words, data_infrastructure):**
```
The choice between a data warehouse and a data lake is not primarily a technical decision — it is a decision about who your data consumers are and how reliably they can tolerate schema changes.\n\nA data warehouse imposes structure at write time. Every dataset must conform to a predefined schema before it enters the warehouse. This constraint is a feature: analysts and BI tools can query data confidently because the schema is enforced and documented. The tradeoff is that onboarding a new data source requires schema design work upfront, which slows iteration and creates a bottleneck when data engineering capacity is limited.\n\nA data lake defers structure to read time. Raw data lands in object storage and consumers apply their own schema when querying. This makes ingestion fast and flexible. The cost is that the lake accumulates data whose quality and meaning are only as good as what each producing system happened to emit. Without disciplined curation, data lakes become data swamps: technically queryable but practically unusable because no one agrees on what the fields mean.\n\nMost mature data platforms end up with both. Raw and semi-structured data lands in the lake. High-value, high-confidence datasets are promoted to warehouse tables through transformation pipelines. The lake handles exploratory analysis; the warehouse handles reporting where consistency matters.\n\nThe decision is often driven by team size. A three-person analytics team can maintain the curation discipline that a lake requires. A team of fifteen analysts with varying SQL proficiency needs the guardrails of a warehouse to avoid spending half their time debugging data quality. Organizations that skip the warehouse layer and rely entirely on a lake typically revisit that decision within eighteen months as analyst headcount grows.
```

Generate the remaining 28 passages following this format. Ensure each passage has a distinct first sentence, covers a different angle within its topic area, and falls between 300–800 words.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_corpus.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add corpus/tasks.json tests/test_corpus.py
git commit -m "feat: task corpus — 30 synthetic passages across 10 topic areas"
```

---

### Task 4: Prompt Builder

**Files:**
- Create: `src/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompts.py`:

```python
import pytest
from src.prompts import build_prompt

CONTENT = "Acme Corp reported record revenue in Q3."


def test_bare_prompt():
    assert build_prompt(CONTENT, "bare") == f"Summarize the following text:\n\n{CONTENT}"


def test_polite_prompt():
    assert build_prompt(CONTENT, "polite") == f"Please summarize the following text:\n\n{CONTENT}"


def test_overly_polite_contains_framing_and_content():
    result = build_prompt(CONTENT, "overly_polite")
    assert result.startswith("Hey, when you get a chance")
    assert CONTENT in result


def test_content_identical_across_variants():
    from src.schema import VARIANTS
    for variant in VARIANTS:
        assert CONTENT in build_prompt(CONTENT, variant)


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_prompt(CONTENT, "rude")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_prompts.py -v
```

Expected: `ImportError` — `src.prompts` does not exist.

- [ ] **Step 3: Create `src/prompts.py`**

```python
PROMPT_TEMPLATES = {
    "bare": "Summarize the following text:\n\n{content}",
    "polite": "Please summarize the following text:\n\n{content}",
    "overly_polite": (
        "Hey, when you get a chance, could you please summarize the following "
        "text for me? Thank you.\n\n{content}"
    ),
}


def build_prompt(content: str, variant: str) -> str:
    if variant not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {list(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[variant].format(content=content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_prompts.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prompts.py tests/test_prompts.py
git commit -m "feat: prompt builder — three variant templates with validation"
```

---

### Task 5: Cost Calculator

**Files:**
- Create: `src/costs.py`
- Create: `tests/test_costs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_costs.py`:

```python
from src.costs import calculate_cost, MODEL_PRICING


def test_all_models_have_pricing():
    from src.schema import MODELS
    for model in MODELS:
        assert model in MODEL_PRICING


def test_zero_tokens_is_zero_cost():
    assert calculate_cost("claude-opus-4-8", 0, 0, 0) == 0.0


def test_opus_1m_input_tokens():
    cost = calculate_cost("claude-opus-4-8", 1_000_000, 0, 0)
    assert abs(cost - 5.00) < 0.0001


def test_opus_1m_output_tokens():
    cost = calculate_cost("claude-opus-4-8", 0, 1_000_000, 0)
    assert abs(cost - 25.00) < 0.0001


def test_reasoning_tokens_billed_at_output_rate():
    output_cost = calculate_cost("claude-opus-4-8", 0, 1_000_000, 0)
    reasoning_cost = calculate_cost("claude-opus-4-8", 0, 0, 1_000_000)
    assert abs(output_cost - reasoning_cost) < 0.0001


def test_gpt55_output_rate_higher_than_opus():
    claude = calculate_cost("claude-opus-4-8", 0, 1_000_000, 0)
    gpt = calculate_cost("gpt-5.5", 0, 1_000_000, 0)
    assert gpt > claude


def test_sonnet_pricing():
    # $3/1M input + $15/1M output = $18 for 1M each
    cost = calculate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 0)
    assert abs(cost - 18.00) < 0.0001
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_costs.py -v
```

Expected: `ImportError` — `src.costs` does not exist.

- [ ] **Step 3: Create `src/costs.py`**

```python
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8":   {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "gpt-5.5":           {"input": 5.00,  "output": 30.00},
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
) -> float:
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    # Reasoning tokens are billed at the output rate on all three models
    output_cost = ((output_tokens + reasoning_tokens) / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_costs.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/costs.py tests/test_costs.py
git commit -m "feat: cost calculator — per-model pricing, reasoning tokens at output rate"
```

---

### Task 6: Anthropic Wrapper

**Files:**
- Create: `src/models/anthropic.py`
- Create: `tests/test_models/test_anthropic.py`

> **Note on `reasoning_tokens`:** The wrapper reads `usage.thinking_tokens` via `getattr(..., None)`. If the installed SDK version does not expose this field, the value defaults to 0. After running a real trial, print `response.usage` to verify the actual field names and update accordingly.

- [ ] **Step 1: Write failing tests**

Create `tests/test_models/test_anthropic.py`:

```python
from unittest.mock import MagicMock, patch
from src.models.anthropic import run_trial


def make_mock_response(
    input_tokens=100,
    output_tokens=50,
    thinking_tokens=200,
    cache_read_tokens=0,
    response_text="This is a summary.",
):
    mock = MagicMock()
    mock.content = [
        MagicMock(type="thinking", thinking="[internal reasoning]"),
        MagicMock(type="text", text=response_text),
    ]
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    mock.usage.thinking_tokens = thinking_tokens
    mock.usage.cache_read_input_tokens = cache_read_tokens
    return mock


@patch("src.models.anthropic.anthropic.Anthropic")
def test_returns_normalized_usage(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.messages.create.return_value = make_mock_response()

    result = run_trial("Summarize this:\n\nhello world", model="claude-opus-4-8")

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["reasoning_tokens"] == 200
    assert result["cached_tokens"] == 0
    assert result["total_tokens"] == 350
    assert result["response_text"] == "This is a summary."
    assert isinstance(result["latency_ms"], int)


@patch("src.models.anthropic.anthropic.Anthropic")
def test_extracts_text_block(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.messages.create.return_value = make_mock_response(response_text="Short summary.")

    result = run_trial("Please summarize:\n\ncontent", model="claude-sonnet-4-6")

    assert result["response_text"] == "Short summary."


@patch("src.models.anthropic.anthropic.Anthropic")
def test_cache_hit_captured(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.messages.create.return_value = make_mock_response(cache_read_tokens=80)

    result = run_trial("Summarize:\n\ntext", model="claude-opus-4-8")

    assert result["cached_tokens"] == 80


@patch("src.models.anthropic.anthropic.Anthropic")
def test_uses_adaptive_thinking_no_temperature(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.messages.create.return_value = make_mock_response()

    run_trial("Summarize:\n\ntext", model="claude-opus-4-8")

    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["max_tokens"] == 4096
    assert "temperature" not in kwargs


@patch("src.models.anthropic.anthropic.Anthropic")
def test_missing_thinking_tokens_field_defaults_to_zero(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    response = make_mock_response()
    del response.usage.thinking_tokens
    mock_client.messages.create.return_value = response

    result = run_trial("Summarize:\n\ntext", model="claude-opus-4-8")

    assert result["reasoning_tokens"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_models/test_anthropic.py -v
```

Expected: `ImportError` — `src.models.anthropic` does not exist.

- [ ] **Step 3: Create `src/models/anthropic.py`**

```python
import time
import anthropic


def run_trial(prompt: str, model: str = "claude-opus-4-8") -> dict:
    client = anthropic.Anthropic()

    start = time.monotonic()
    response = client.messages.create(
        model=model,
        thinking={"type": "adaptive"},
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    text_blocks = [b for b in response.content if b.type == "text"]
    response_text = text_blocks[0].text if text_blocks else ""

    usage = response.usage
    reasoning_tokens = getattr(usage, "thinking_tokens", None) or 0
    cached_tokens = getattr(usage, "cache_read_input_tokens", None) or 0

    return {
        "response_text": response_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens + reasoning_tokens,
        "latency_ms": latency_ms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_models/test_anthropic.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/anthropic.py tests/test_models/test_anthropic.py
git commit -m "feat: anthropic wrapper — opus 4.8 + sonnet 4.6, adaptive thinking, usage normalization"
```

---

### Task 7: OpenAI Wrapper

**Files:**
- Create: `src/models/openai.py`
- Create: `tests/test_models/test_openai.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models/test_openai.py`:

```python
from unittest.mock import MagicMock, patch
from src.models.openai import run_trial


def make_mock_response(
    prompt_tokens=100,
    completion_tokens=250,
    reasoning_tokens=200,
    cached_tokens=0,
    response_text="This is a summary.",
):
    mock = MagicMock()
    mock.choices[0].message.content = response_text
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.usage.total_tokens = prompt_tokens + completion_tokens
    mock.usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
    mock.usage.prompt_tokens_details.cached_tokens = cached_tokens
    return mock


@patch("src.models.openai.OpenAI")
def test_returns_normalized_usage(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = make_mock_response()

    result = run_trial("Summarize this:\n\nhello world")

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50   # completion(250) - reasoning(200)
    assert result["reasoning_tokens"] == 200
    assert result["cached_tokens"] == 0
    assert result["total_tokens"] == 350   # input + output + reasoning
    assert result["response_text"] == "This is a summary."
    assert isinstance(result["latency_ms"], int)


@patch("src.models.openai.OpenAI")
def test_cache_hit_captured(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = make_mock_response(cached_tokens=60)

    result = run_trial("Summarize:\n\ntext")

    assert result["cached_tokens"] == 60


@patch("src.models.openai.OpenAI")
def test_model_and_no_reasoning_effort_override(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = make_mock_response()

    run_trial("Summarize:\n\ntext")

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["max_completion_tokens"] == 4096
    # reasoning_effort intentionally absent — let the model reason naturally
    assert "reasoning_effort" not in kwargs
    assert "temperature" not in kwargs


@patch("src.models.openai.OpenAI")
def test_null_details_defaults_to_zero(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    response = make_mock_response()
    response.usage.completion_tokens_details = None
    response.usage.prompt_tokens_details = None
    mock_client.chat.completions.create.return_value = response

    result = run_trial("Summarize:\n\ntext")

    assert result["reasoning_tokens"] == 0
    assert result["cached_tokens"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_models/test_openai.py -v
```

Expected: `ImportError` — `src.models.openai` does not exist.

- [ ] **Step 3: Create `src/models/openai.py`**

```python
import time
from openai import OpenAI


def run_trial(prompt: str) -> dict:
    client = OpenAI()

    start = time.monotonic()
    response = client.chat.completions.create(
        model="gpt-5.5",
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    usage = response.usage
    completion_details = usage.completion_tokens_details
    reasoning_tokens = completion_details.reasoning_tokens if completion_details else 0

    prompt_details = usage.prompt_tokens_details
    cached_tokens = prompt_details.cached_tokens if prompt_details else 0

    # completion_tokens includes reasoning; output_tokens is the visible text only
    output_tokens = usage.completion_tokens - reasoning_tokens

    return {
        "response_text": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": usage.prompt_tokens + output_tokens + reasoning_tokens,
        "latency_ms": latency_ms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_models/test_openai.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/openai.py tests/test_models/test_openai.py
git commit -m "feat: openai wrapper — gpt-5.5, reasoning token extraction, usage normalization"
```

---

### Task 8: Runner

**Files:**
- Create: `src/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner.py`:

```python
import json
from pathlib import Path
from src.runner import (
    load_completed_trials,
    generate_all_trials,
    build_trial_record,
    append_result,
)
from src.schema import MODELS, VARIANTS, REPS

SAMPLE_CORPUS = [
    {"task_id": f"T{i:02d}", "title": f"Task {i}", "topic": "test", "content": f"Content {i}."}
    for i in range(1, 4)
]


def test_generate_all_trials_count():
    trials = generate_all_trials(SAMPLE_CORPUS)
    assert len(trials) == len(SAMPLE_CORPUS) * len(MODELS) * len(VARIANTS) * REPS


def test_generate_all_trials_structure():
    task, model, variant, rep = generate_all_trials(SAMPLE_CORPUS)[0]
    assert task in SAMPLE_CORPUS
    assert model in MODELS
    assert variant in VARIANTS
    assert 1 <= rep <= REPS


def test_load_completed_trials_empty(tmp_path):
    f = tmp_path / "raw.jsonl"
    f.write_text("")
    assert load_completed_trials(f) == set()


def test_load_completed_trials_skips_malformed(tmp_path):
    f = tmp_path / "raw.jsonl"
    f.write_text('{"trial_id": "a|b|c|1"}\n{bad}\n{"trial_id": "d|e|f|2"}\n')
    assert load_completed_trials(f) == {"a|b|c|1", "d|e|f|2"}


def test_load_completed_trials_missing_file(tmp_path):
    assert load_completed_trials(tmp_path / "nope.jsonl") == set()


def test_append_result_creates_and_accumulates(tmp_path):
    f = tmp_path / "raw.jsonl"
    append_result({"trial_id": "a"}, f)
    append_result({"trial_id": "b"}, f)
    lines = [json.loads(l) for l in f.read_text().strip().split("\n")]
    assert [r["trial_id"] for r in lines] == ["a", "b"]


def test_build_trial_record_fields():
    task = {"task_id": "T01", "title": "T", "topic": "t", "content": "Some content here."}
    api_result = {
        "response_text": "A summary.",
        "input_tokens": 100,
        "output_tokens": 40,
        "reasoning_tokens": 250,
        "cached_tokens": 0,
        "total_tokens": 390,
        "latency_ms": 1200,
    }
    record = build_trial_record(task, "claude-opus-4-8", "bare", 1, api_result)
    assert record["trial_id"] == "claude-opus-4-8|T01|bare|1"
    assert record["reasoning_tokens"] == 250
    assert isinstance(record["cost_usd"], float)
    assert record["cost_usd"] > 0
    assert "timestamp" in record
    assert record["quality"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_runner.py -v
```

Expected: `ImportError` — `src.runner` does not exist.

- [ ] **Step 3: Create `src/runner.py`**

```python
import json
import random
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.schema import MODELS, VARIANTS, REPS, make_trial_id
from src.prompts import build_prompt
from src.costs import calculate_cost

load_dotenv()

CORPUS_PATH = Path("corpus/tasks.json")
RESULTS_PATH = Path("results/raw.jsonl")
RUN_LOG_PATH = Path("results/run_log.jsonl")


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def load_completed_trials(path: Path = RESULTS_PATH) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def generate_all_trials(corpus: list[dict]) -> list[tuple]:
    return [
        (task, model, variant, rep)
        for task in corpus
        for model in MODELS
        for variant in VARIANTS
        for rep in range(1, REPS + 1)
    ]


def build_trial_record(
    task: dict, model: str, variant: str, rep: int, api_result: dict
) -> dict:
    return {
        "trial_id": make_trial_id(model, task["task_id"], variant, rep),
        "model": model,
        "task_id": task["task_id"],
        "variant": variant,
        "rep": rep,
        "prompt_text": build_prompt(task["content"], variant),
        "response_text": api_result["response_text"],
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "reasoning_tokens": api_result["reasoning_tokens"],
        "cached_tokens": api_result["cached_tokens"],
        "total_tokens": api_result["total_tokens"],
        "cost_usd": calculate_cost(
            model,
            api_result["input_tokens"],
            api_result["output_tokens"],
            api_result["reasoning_tokens"],
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": api_result["latency_ms"],
        "quality": None,
    }


def append_result(record: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _call_api(model: str, prompt: str) -> dict:
    if model.startswith("claude"):
        from src.models.anthropic import run_trial
        return run_trial(prompt, model=model)
    from src.models.openai import run_trial
    return run_trial(prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    corpus = load_corpus()
    completed = load_completed_trials()
    all_trials = generate_all_trials(corpus)

    if args.model != "all":
        all_trials = [(t, m, v, r) for t, m, v, r in all_trials if m == args.model]

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)
    random.shuffle(all_trials)

    pending = [
        (t, m, v, r) for t, m, v, r in all_trials
        if make_trial_id(m, t["task_id"], v, r) not in completed
    ]

    print(f"Total: {len(all_trials)}  Completed: {len(completed)}  Pending: {len(pending)}  Seed: {seed}")

    if args.dry_run:
        print("Dry run — no API calls made.")
        return

    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a") as f:
        f.write(json.dumps({"seed": seed, "pending": len(pending),
                             "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")

    errors = 0
    for task, model, variant, rep in tqdm(pending, desc="Running trials"):
        prompt = build_prompt(task["content"], variant)
        try:
            api_result = _call_api(model, prompt)
        except Exception as e:
            tid = make_trial_id(model, task["task_id"], variant, rep)
            print(f"\nERROR {tid}: {e}", file=sys.stderr)
            errors += 1
            continue
        append_result(build_trial_record(task, model, variant, rep, api_result))

    print(f"\nDone. Errors: {errors}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_runner.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/runner.py tests/test_runner.py
git commit -m "feat: runner — sequential execution, resumption, randomized order, dry-run flag"
```

---

### Task 9: Analyzer

**Files:**
- Create: `tests/fixtures/sample_results.jsonl`
- Create: `src/analyze.py`
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Create fixture data**

Create `tests/fixtures/sample_results.jsonl` — 9 records (1 task × 3 variants × 3 reps, one cache hit on polite|rep3):

```jsonl
{"trial_id": "claude-opus-4-8|T01|bare|1", "model": "claude-opus-4-8", "task_id": "T01", "variant": "bare", "rep": 1, "prompt_text": "Summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 50, "output_tokens": 20, "reasoning_tokens": 100, "cached_tokens": 0, "total_tokens": 170, "cost_usd": 0.0028, "timestamp": "2026-06-05T12:00:00Z", "latency_ms": 800, "quality": null}
{"trial_id": "claude-opus-4-8|T01|bare|2", "model": "claude-opus-4-8", "task_id": "T01", "variant": "bare", "rep": 2, "prompt_text": "Summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 50, "output_tokens": 22, "reasoning_tokens": 120, "cached_tokens": 0, "total_tokens": 192, "cost_usd": 0.0031, "timestamp": "2026-06-05T12:01:00Z", "latency_ms": 900, "quality": null}
{"trial_id": "claude-opus-4-8|T01|bare|3", "model": "claude-opus-4-8", "task_id": "T01", "variant": "bare", "rep": 3, "prompt_text": "Summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 50, "output_tokens": 18, "reasoning_tokens": 90, "cached_tokens": 0, "total_tokens": 158, "cost_usd": 0.0025, "timestamp": "2026-06-05T12:02:00Z", "latency_ms": 750, "quality": null}
{"trial_id": "claude-opus-4-8|T01|polite|1", "model": "claude-opus-4-8", "task_id": "T01", "variant": "polite", "rep": 1, "prompt_text": "Please summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 51, "output_tokens": 21, "reasoning_tokens": 350, "cached_tokens": 0, "total_tokens": 422, "cost_usd": 0.0064, "timestamp": "2026-06-05T12:03:00Z", "latency_ms": 1200, "quality": null}
{"trial_id": "claude-opus-4-8|T01|polite|2", "model": "claude-opus-4-8", "task_id": "T01", "variant": "polite", "rep": 2, "prompt_text": "Please summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 51, "output_tokens": 25, "reasoning_tokens": 380, "cached_tokens": 0, "total_tokens": 456, "cost_usd": 0.0069, "timestamp": "2026-06-05T12:04:00Z", "latency_ms": 1300, "quality": null}
{"trial_id": "claude-opus-4-8|T01|polite|3", "model": "claude-opus-4-8", "task_id": "T01", "variant": "polite", "rep": 3, "prompt_text": "Please summarize:\n\nContent.", "response_text": "A summary.", "input_tokens": 51, "output_tokens": 19, "reasoning_tokens": 320, "cached_tokens": 60, "total_tokens": 390, "cost_usd": 0.0058, "timestamp": "2026-06-05T12:05:00Z", "latency_ms": 950, "quality": null}
{"trial_id": "claude-opus-4-8|T01|overly_polite|1", "model": "claude-opus-4-8", "task_id": "T01", "variant": "overly_polite", "rep": 1, "prompt_text": "Hey, when you get a chance...\n\nContent.", "response_text": "A summary.", "input_tokens": 60, "output_tokens": 28, "reasoning_tokens": 600, "cached_tokens": 0, "total_tokens": 688, "cost_usd": 0.0103, "timestamp": "2026-06-05T12:06:00Z", "latency_ms": 1800, "quality": null}
{"trial_id": "claude-opus-4-8|T01|overly_polite|2", "model": "claude-opus-4-8", "task_id": "T01", "variant": "overly_polite", "rep": 2, "prompt_text": "Hey, when you get a chance...\n\nContent.", "response_text": "A summary.", "input_tokens": 60, "output_tokens": 30, "reasoning_tokens": 580, "cached_tokens": 0, "total_tokens": 670, "cost_usd": 0.0100, "timestamp": "2026-06-05T12:07:00Z", "latency_ms": 1750, "quality": null}
{"trial_id": "claude-opus-4-8|T01|overly_polite|3", "model": "claude-opus-4-8", "task_id": "T01", "variant": "overly_polite", "rep": 3, "prompt_text": "Hey, when you get a chance...\n\nContent.", "response_text": "A summary.", "input_tokens": 60, "output_tokens": 26, "reasoning_tokens": 620, "cached_tokens": 0, "total_tokens": 706, "cost_usd": 0.0106, "timestamp": "2026-06-05T12:08:00Z", "latency_ms": 1900, "quality": null}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_analyze.py`:

```python
import pandas as pd
from pathlib import Path
from src.analyze import load_results, make_clean, save_trials_csv, save_summary_csv

FIXTURE = Path("tests/fixtures/sample_results.jsonl")


def test_load_results_count():
    assert len(load_results(FIXTURE)) == 9


def test_load_results_has_required_columns():
    df = load_results(FIXTURE)
    for col in ["trial_id", "model", "variant", "reasoning_tokens", "cached_tokens"]:
        assert col in df.columns


def test_make_clean_excludes_cache_hits():
    df = load_results(FIXTURE)
    clean = make_clean(df)
    assert len(clean) == 8
    assert (clean["cached_tokens"] == 0).all()


def test_save_trials_csv_creates_both_files(tmp_path):
    df = load_results(FIXTURE)
    save_trials_csv(df, tmp_path)
    assert (tmp_path / "trials_raw.csv").exists()
    assert (tmp_path / "trials_clean.csv").exists()


def test_trials_csv_row_counts(tmp_path):
    df = load_results(FIXTURE)
    save_trials_csv(df, tmp_path)
    assert len(pd.read_csv(tmp_path / "trials_raw.csv")) == 9
    assert len(pd.read_csv(tmp_path / "trials_clean.csv")) == 8


def test_summary_has_all_variants(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["variant"]) == {"bare", "polite", "overly_polite"}


def test_summary_has_reasoning_stats(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert "avg_reasoning_tokens" in summary.columns
    assert "std_reasoning_tokens" in summary.columns


def test_overly_polite_has_most_reasoning(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    bare = summary[summary["variant"] == "bare"]["avg_reasoning_tokens"].values[0]
    overly = summary[summary["variant"] == "overly_polite"]["avg_reasoning_tokens"].values[0]
    assert overly > bare
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_analyze.py -v
```

Expected: `ImportError` — `src.analyze` does not exist.

- [ ] **Step 4: Create `src/analyze.py`**

```python
import json
import argparse
import pandas as pd
from pathlib import Path

RESULTS_PATH = Path("results/raw.jsonl")
ANALYSIS_DIR = Path("results/analysis")

TRIAL_COLUMNS = [
    "trial_id", "model", "task_id", "variant", "rep",
    "input_tokens", "output_tokens", "reasoning_tokens",
    "cached_tokens", "total_tokens", "cost_usd", "latency_ms", "timestamp",
]


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def make_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["cached_tokens"] == 0].copy()


def save_trials_csv(df: pd.DataFrame, output_dir: Path = ANALYSIS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in TRIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(output_dir / "trials_raw.csv", index=False)
    make_clean(df)[cols].to_csv(output_dir / "trials_clean.csv", index=False)


def save_summary_csv(df: pd.DataFrame, output_dir: Path = ANALYSIS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = make_clean(df)
    summary = (
        clean.groupby(["model", "variant"])
        .agg(
            n_trials=("trial_id", "count"),
            avg_reasoning_tokens=("reasoning_tokens", "mean"),
            std_reasoning_tokens=("reasoning_tokens", "std"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
            cache_hit_rate=("cached_tokens", lambda x: (x > 0).mean()),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    df = load_results(args.results)
    cache_hits = (df["cached_tokens"] > 0).sum()
    print(f"Loaded {len(df)} trials. Cache hits: {cache_hits} ({cache_hits / len(df) * 100:.1f}%)")

    save_trials_csv(df, args.output)
    save_summary_csv(df, args.output)

    print(f"\nWritten to {args.output}/")
    print("  trials_raw.csv    — all trials")
    print("  trials_clean.csv  — cache-excluded")
    print("  summary.csv       — per-model × variant aggregates")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/repos/politeness-experiment && pytest tests/test_analyze.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run the full test suite**

```bash
cd ~/repos/politeness-experiment && pytest -v
```

Expected: all tests PASS across all modules. Count should be 36+ tests.

- [ ] **Step 7: Commit**

```bash
git add src/analyze.py tests/test_analyze.py tests/fixtures/sample_results.jsonl
git commit -m "feat: analyzer — JSONL to long-format CSVs, cache-clean view, per-variant summary"
```

---

## Self-Review

**Spec coverage:**
- ✅ 1,350 trials — 30 × 3 × 5 × 3 in `generate_all_trials`
- ✅ Three prompt variants — `VARIANTS` constant + `build_prompt`
- ✅ Three models — `MODELS` constant
- ✅ Adaptive thinking on Claude, no `temperature` parameter — Task 6
- ✅ Default reasoning effort on GPT-5.5 (no override) — Task 7
- ✅ Reasoning tokens as primary metric — `reasoning_tokens` in schema, wrapper, record
- ✅ JSONL append-only storage — `append_result`
- ✅ Resumability — `load_completed_trials` + trial ID dedup
- ✅ Randomized order with logged seed — `main()` in runner
- ✅ Cache detection — `cached_tokens` field in both wrappers
- ✅ Dual view (raw + clean) — `make_clean`, `save_trials_csv`
- ✅ Long-format CSV for visualization — `trials_raw.csv`, `trials_clean.csv`
- ✅ Summary CSV — `save_summary_csv` with avg/std per model × variant
- ✅ Per-trial cost — `calculate_cost` called in `build_trial_record`
- ✅ `--dry-run` flag
- ✅ `--model` filter flag
- ✅ Seed logged to `run_log.jsonl`

**Placeholder scan:** No TBDs, no "similar to above," no steps without code.

**Type consistency:**
- `make_trial_id(model, task_id, variant, rep)` — defined Task 2, called Task 8 ✅
- `build_prompt(content, variant)` — defined Task 4, called Task 8 ✅
- `calculate_cost(model, input_tokens, output_tokens, reasoning_tokens)` — defined Task 5, called Task 8 ✅
- `load_results(path)`, `make_clean(df)`, `save_trials_csv(df, dir)`, `save_summary_csv(df, dir)` — defined and tested consistently in Task 9 ✅
- `load_completed_trials(path)`, `generate_all_trials(corpus)`, `build_trial_record(task, model, variant, rep, api_result)`, `append_result(record, path)` — all defined and called with matching signatures ✅
