import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.grade import GROUND_TRUTHS, grade_trial, load_graded


def _make_haiku_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


@patch("src.grade.anthropic.Anthropic")
def test_grade_correct(mock_class):
    client = MagicMock()
    mock_class.return_value = client
    client.messages.create.return_value = _make_haiku_response(
        "CORRECT\nThe response states 480 miles as the final answer."
    )

    result = grade_trial("Train problem...", "R01", "The answer is 480 miles.", client)

    assert result["correct"] is True
    assert "480 miles" in result["explanation"]


@patch("src.grade.anthropic.Anthropic")
def test_grade_incorrect(mock_class):
    client = MagicMock()
    mock_class.return_value = client
    client.messages.create.return_value = _make_haiku_response(
        "INCORRECT\nThe response says 240 miles, not 480."
    )

    result = grade_trial("Train problem...", "R01", "The answer is 240 miles.", client)

    assert result["correct"] is False
    assert result["explanation"]


def test_grade_unknown_task():
    client = MagicMock()
    result = grade_trial("...", "R99", "some response", client)

    assert result["correct"] is None
    assert "R99" in result["explanation"]
    client.messages.create.assert_not_called()


def test_load_graded_empty(tmp_path):
    assert load_graded(tmp_path / "nonexistent.jsonl") == set()


def test_load_graded_reads_trial_ids(tmp_path):
    f = tmp_path / "graded.jsonl"
    f.write_text(
        json.dumps({"trial_id": "m|R01|bare|0", "quality": {"correct": True}}) + "\n"
        + json.dumps({"trial_id": "m|R01|polite|0", "quality": {"correct": False}}) + "\n"
    )

    result = load_graded(f)

    assert result == {"m|R01|bare|0", "m|R01|polite|0"}


def test_load_graded_skips_corrupt_lines(tmp_path):
    f = tmp_path / "graded.jsonl"
    f.write_text(
        '{"trial_id": "good|id"}\n'
        "not json\n"
        '{"no_trial_id": true}\n'
    )

    result = load_graded(f)

    assert result == {"good|id"}


def test_ground_truths_covers_r01_to_r10():
    for i in range(1, 11):
        key = f"R{i:02d}"
        assert key in GROUND_TRUTHS, f"Missing ground truth for {key}"
        assert GROUND_TRUTHS[key]["answer"]
        assert GROUND_TRUTHS[key]["detail"]


@patch("src.grade.anthropic.Anthropic")
def test_grade_truncates_long_response(mock_class):
    client = MagicMock()
    client.messages.create.return_value = _make_haiku_response("CORRECT\nOk.")
    long_response = "x" * 10000

    grade_trial("problem", "R01", long_response, client)

    call_kwargs = client.messages.create.call_args[1]
    prompt_sent = call_kwargs["messages"][0]["content"]
    # response was capped at 4000 chars before being embedded in prompt
    assert "x" * 4001 not in prompt_sent
