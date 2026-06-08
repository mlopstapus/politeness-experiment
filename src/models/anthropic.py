import time
import anthropic


def run_trial(prompt: str, model: str = "claude-opus-4-8") -> dict:
    client = anthropic.Anthropic()

    start = time.monotonic()
    response = client.messages.create(
        model=model,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    text_blocks = [b for b in response.content if b.type == "text"]
    response_text = text_blocks[0].text if text_blocks else ""

    usage = response.usage
    # thinking_tokens lives in output_tokens_details; output_tokens includes thinking
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = (details.thinking_tokens if details else 0) or 0
    # visible output only (Claude's output_tokens includes thinking tokens)
    output_tokens = usage.output_tokens - reasoning_tokens
    cached_tokens = getattr(usage, "cache_read_input_tokens", None) or 0

    return {
        "response_text": response_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": usage.input_tokens + output_tokens + reasoning_tokens,
        "latency_ms": latency_ms,
    }
