import pandas as pd
from pathlib import Path
from src.analyze import load_results, make_clean, save_trials_csv, save_summary_csv

FIXTURE = Path("tests/fixtures/sample_results.jsonl")


def test_load_results_count():
    assert len(load_results(FIXTURE)) == 9


def test_load_results_has_required_columns():
    df = load_results(FIXTURE)
    for col in ["trial_id", "model", "variant", "reasoning_tokens", "cached_tokens"]:
        assert col in df.columns


def test_make_clean_excludes_cache_hits():
    df = load_results(FIXTURE)
    clean = make_clean(df)
    assert len(clean) == 8
    assert (clean["cached_tokens"] == 0).all()


def test_save_trials_csv_creates_both_files(tmp_path):
    df = load_results(FIXTURE)
    save_trials_csv(df, tmp_path)
    assert (tmp_path / "trials_raw.csv").exists()
    assert (tmp_path / "trials_clean.csv").exists()


def test_trials_csv_row_counts(tmp_path):
    df = load_results(FIXTURE)
    save_trials_csv(df, tmp_path)
    assert len(pd.read_csv(tmp_path / "trials_raw.csv")) == 9
    assert len(pd.read_csv(tmp_path / "trials_clean.csv")) == 8


def test_summary_has_all_variants(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["variant"]) == {"bare", "polite", "overly_polite"}


def test_summary_has_reasoning_stats(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert "avg_reasoning_tokens" in summary.columns
    assert "std_reasoning_tokens" in summary.columns


def test_overly_polite_has_most_reasoning(tmp_path):
    df = load_results(FIXTURE)
    save_summary_csv(df, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv")
    bare = summary[summary["variant"] == "bare"]["avg_reasoning_tokens"].values[0]
    overly = summary[summary["variant"] == "overly_polite"]["avg_reasoning_tokens"].values[0]
    assert overly > bare
