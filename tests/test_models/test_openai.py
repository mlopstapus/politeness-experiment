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
    assert kwargs["reasoning_effort"] == "high"
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
