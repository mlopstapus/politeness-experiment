from unittest.mock import MagicMock, patch
from src.models.anthropic import run_trial


def make_mock_response(
    input_tokens=100,
    visible_output_tokens=50,
    thinking_tokens=200,
    cache_read_tokens=0,
    response_text="This is a summary.",
):
    # Real API: usage.output_tokens includes thinking tokens
    total_output = visible_output_tokens + thinking_tokens
    mock = MagicMock()
    mock.content = [
        MagicMock(type="thinking", thinking="[internal reasoning]"),
        MagicMock(type="text", text=response_text),
    ]
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = total_output
    mock.usage.output_tokens_details.thinking_tokens = thinking_tokens
    mock.usage.cache_read_input_tokens = cache_read_tokens
    return mock


@patch("src.models.anthropic.anthropic.Anthropic")
def test_returns_normalized_usage(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_client.messages.create.return_value = make_mock_response()

    result = run_trial("Summarize this:\n\nhello world", model="claude-opus-4-8")

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50       # visible only
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
    assert kwargs["max_tokens"] == 16000
    assert "temperature" not in kwargs


@patch("src.models.anthropic.anthropic.Anthropic")
def test_missing_thinking_details_defaults_to_zero(mock_class):
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    response = make_mock_response()
    response.usage.output_tokens_details = None
    mock_client.messages.create.return_value = response

    result = run_trial("Summarize:\n\ntext", model="claude-opus-4-8")

    assert result["reasoning_tokens"] == 0
