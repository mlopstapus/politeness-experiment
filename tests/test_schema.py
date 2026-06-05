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
