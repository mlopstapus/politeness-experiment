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
