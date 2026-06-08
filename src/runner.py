import json
import random
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.schema import MODELS, VARIANTS, REPS, make_trial_id
from src.prompts import build_prompt
from src.costs import calculate_cost

ANTHROPIC_MODELS = [m for m in MODELS if m.startswith("claude")]
OPENAI_MODELS = [m for m in MODELS if not m.startswith("claude")]

CORPUS_PATH = Path("corpus/tasks.json")
RESULTS_PATH = Path("results/raw.jsonl")
RUN_LOG_PATH = Path("results/run_log.jsonl")


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"{path} not found — run from the project root")
    with path.open() as f:
        return json.load(f)


def load_completed_trials(path: Path = RESULTS_PATH) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def generate_all_trials(corpus: list[dict]) -> list[tuple]:
    return [
        (task, model, variant, rep)
        for task in corpus
        for model in MODELS
        for variant in VARIANTS
        for rep in range(1, REPS + 1)
    ]


def build_trial_record(
    task: dict, model: str, variant: str, rep: int, api_result: dict, prompt: str
) -> dict:
    return {
        "trial_id": make_trial_id(model, task["task_id"], variant, rep),
        "model": model,
        "task_id": task["task_id"],
        "variant": variant,
        "rep": rep,
        "prompt_text": prompt,
        "response_text": api_result["response_text"],
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "reasoning_tokens": api_result["reasoning_tokens"],
        "cached_tokens": api_result["cached_tokens"],
        "total_tokens": api_result["total_tokens"],
        "cost_usd": calculate_cost(
            model,
            api_result["input_tokens"],
            api_result["output_tokens"],
            api_result["reasoning_tokens"],
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": api_result["latency_ms"],
        "quality": None,
    }


def append_result(record: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _call_api(model: str, prompt: str) -> dict:
    if model in ANTHROPIC_MODELS:
        from src.models.anthropic import run_trial
        return run_trial(prompt, model=model)
    if model in OPENAI_MODELS:
        from src.models.openai import run_trial
        return run_trial(prompt)
    raise ValueError(f"Unknown model: {model!r}. Must be one of {MODELS}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = load_corpus()
    completed = load_completed_trials()
    all_trials = generate_all_trials(corpus)

    if args.model != "all":
        all_trials = [(t, m, v, r) for t, m, v, r in all_trials if m == args.model]

    seed = args.seed
    random.seed(seed)
    random.shuffle(all_trials)

    pending = [
        (t, m, v, r) for t, m, v, r in all_trials
        if make_trial_id(m, t["task_id"], v, r) not in completed
    ]

    scoped_completed = len(all_trials) - len(pending)
    print(f"Total: {len(all_trials)}  Completed: {scoped_completed}  Pending: {len(pending)}  Seed: {seed}")

    if args.dry_run:
        print("Dry run — no API calls made.")
        return

    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a") as f:
        f.write(json.dumps({"seed": seed, "pending": len(pending),
                             "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")

    errors = 0
    for task, model, variant, rep in tqdm(pending, desc="Running trials"):
        prompt = build_prompt(task["content"], variant)
        try:
            api_result = _call_api(model, prompt)
        except Exception as e:
            tid = make_trial_id(model, task["task_id"], variant, rep)
            print(f"\nERROR {tid}: {e}", file=sys.stderr)
            errors += 1
            continue
        append_result(build_trial_record(task, model, variant, rep, api_result, prompt))

    print(f"\nDone. Errors: {errors}")


if __name__ == "__main__":
    main()
