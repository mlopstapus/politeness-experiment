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
    cost = calculate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 0)
    assert abs(cost - 18.00) < 0.0001
