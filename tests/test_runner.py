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
