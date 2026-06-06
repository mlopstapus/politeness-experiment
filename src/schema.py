from typing import TypedDict

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
    quality: dict | None
