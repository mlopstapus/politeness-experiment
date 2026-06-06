import json
import argparse
import pandas as pd
from pathlib import Path

RESULTS_PATH = Path("results/raw.jsonl")
ANALYSIS_DIR = Path("results/analysis")

TRIAL_COLUMNS = [
    "trial_id", "model", "task_id", "variant", "rep",
    "input_tokens", "output_tokens", "reasoning_tokens",
    "cached_tokens", "total_tokens", "cost_usd", "latency_ms", "timestamp",
]


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def make_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "cached_tokens" not in df.columns:
        return df.copy()
    return df[df["cached_tokens"] == 0].copy()


def save_trials_csv(df: pd.DataFrame, output_dir: Path = ANALYSIS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in TRIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(output_dir / "trials_raw.csv", index=False)
    make_clean(df)[cols].to_csv(output_dir / "trials_clean.csv", index=False)


def save_summary_csv(df: pd.DataFrame, output_dir: Path = ANALYSIS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = make_clean(df)
    # Compute cache_hit_rate from the full df before cleaning
    hit_rates = (
        df.groupby(["model", "variant"])
        .agg(cache_hit_rate=("cached_tokens", lambda x: (x > 0).mean()))
        .reset_index()
    )
    summary = (
        clean.groupby(["model", "variant"])
        .agg(
            n_trials=("trial_id", "count"),
            avg_reasoning_tokens=("reasoning_tokens", "mean"),
            std_reasoning_tokens=("reasoning_tokens", "std"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
        )
        .reset_index()
        .merge(hit_rates, on=["model", "variant"])
    )
    # Round token/count columns to 1dp, cost columns to 6dp to avoid zeroing
    token_cols = [c for c in summary.columns if "tokens" in c or c == "n_trials"]
    cost_cols = [c for c in summary.columns if "cost" in c or "rate" in c]
    summary[token_cols] = summary[token_cols].round(1)
    summary[cost_cols] = summary[cost_cols].round(6)
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    df = load_results(args.results)
    if df.empty:
        print("No trials found.")
        return

    cache_hits = int((df["cached_tokens"] > 0).sum())
    print(f"Loaded {len(df)} trials. Cache hits: {cache_hits} ({cache_hits / len(df) * 100:.1f}%)")

    save_trials_csv(df, args.output)
    save_summary_csv(df, args.output)

    print(f"\nWritten to {args.output}/")
    print("  trials_raw.csv    — all trials")
    print("  trials_clean.csv  — cache-excluded")
    print("  summary.csv       — per-model × variant aggregates")


if __name__ == "__main__":
    main()
