import time
from openai import OpenAI


def run_trial(prompt: str) -> dict:
    client = OpenAI()

    start = time.monotonic()
    response = client.chat.completions.create(
        model="gpt-5.5",
        max_completion_tokens=4096,
        reasoning={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    usage = response.usage
    completion_details = usage.completion_tokens_details
    reasoning_tokens = completion_details.reasoning_tokens if completion_details else 0

    prompt_details = usage.prompt_tokens_details
    cached_tokens = prompt_details.cached_tokens if prompt_details else 0

    # completion_tokens includes reasoning; output_tokens is the visible text only
    output_tokens = max(0, usage.completion_tokens - reasoning_tokens)

    return {
        "response_text": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": usage.prompt_tokens + output_tokens + reasoning_tokens,
        "latency_ms": latency_ms,
    }
