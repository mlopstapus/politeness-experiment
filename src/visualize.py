import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

GRADED_PATH = Path("results/graded_reasoning.jsonl")
FIGURES_DIR = Path("results/figures")

MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5"]
MODEL_LABELS = ["Claude Opus 4.8", "Claude Sonnet 4.6", "GPT-5.5"]
VARIANTS = ["bare", "polite", "overly_polite"]
VARIANT_LABELS = ["Bare", "Polite", "Overly Polite"]

PALETTE = {
    "bare":          "#2C3E50",
    "polite":        "#2980B9",
    "overly_polite": "#85C1E9",
}

STYLE = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
}


def load_data(path: Path = GRADED_PATH) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ── Chart 1: grouped bar — avg reasoning tokens by model × variant ──────────

def chart_reasoning_bars(records: list[dict], out: Path) -> None:
    rstats: dict = defaultdict(list)
    for r in records:
        rstats[(r["model"], r["variant"])].append(r["reasoning_tokens"])

    x = np.arange(len(MODELS))
    width = 0.26
    offsets = [-width, 0, width]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, (variant, label) in enumerate(zip(VARIANTS, VARIANT_LABELS)):
            avgs = [_avg(rstats[(m, variant)]) for m in MODELS]
            bars = ax.bar(x + offsets[i], avgs, width, label=label,
                          color=PALETTE[variant], zorder=3)

            # annotate % change vs bare on Opus only
            if variant != "bare":
                bare_avg = _avg(rstats[(MODELS[0], "bare")])
                pct = (avgs[0] - bare_avg) / bare_avg * 100
                ax.annotate(
                    f"{pct:+.0f}%",
                    xy=(x[0] + offsets[i], avgs[0]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                    color=PALETTE[variant],
                )

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=11)
        ax.set_ylabel("Avg reasoning tokens", fontsize=11)
        ax.set_title("Reasoning tokens by model and prompt politeness",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(framealpha=0.7, fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── Chart 2: box plot — Opus reasoning distribution by variant ───────────────

def chart_opus_boxplot(records: list[dict], out: Path) -> None:
    rstats: dict = defaultdict(list)
    for r in records:
        if r["model"] == "claude-opus-4-8":
            rstats[r["variant"]].append(r["reasoning_tokens"])

    data = [rstats[v] for v in VARIANTS]
    colors = [PALETTE[v] for v in VARIANTS]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markersize=4,
                                        alpha=0.4, linestyle="none"))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for whisker in bp["whiskers"]:
            whisker.set(color="#555", linewidth=1.2)
        for cap in bp["caps"]:
            cap.set(color="#555", linewidth=1.2)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(VARIANT_LABELS, fontsize=11)
        ax.set_ylabel("Reasoning tokens", fontsize=11)
        ax.set_title("Claude Opus 4.8 — reasoning token distribution\nby prompt politeness",
                     fontsize=13, fontweight="bold", pad=12)

        # annotate medians
        for i, vals in enumerate(data, 1):
            med = float(np.median(vals))
            ax.text(i, med + 80, f"{med:.0f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── Chart 3: side-by-side reasoning drop vs flat accuracy (Opus) ────────────

def chart_reasoning_vs_accuracy(records: list[dict], out: Path) -> None:
    rstats: dict = defaultdict(list)
    astats: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        if r["model"] != "claude-opus-4-8":
            continue
        rstats[r["variant"]].append(r["reasoning_tokens"])
        q = (r.get("quality") or {})
        if q.get("correct") is not None:
            astats[r["variant"]]["total"] += 1
            if q["correct"]:
                astats[r["variant"]]["correct"] += 1

    r_avgs = [_avg(rstats[v]) for v in VARIANTS]
    accs   = [astats[v]["correct"] / astats[v]["total"] * 100
              if astats[v]["total"] else 0 for v in VARIANTS]

    x = np.arange(len(VARIANTS))
    colors = [PALETTE[v] for v in VARIANTS]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Claude Opus 4.8 — being polite saves tokens, not accuracy",
                     fontsize=13, fontweight="bold", y=1.01)

        # left: reasoning tokens
        bars = ax1.bar(x, r_avgs, color=colors, zorder=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(VARIANT_LABELS, fontsize=11)
        ax1.set_ylabel("Avg reasoning tokens", fontsize=11)
        ax1.set_title("Reasoning tokens", fontsize=12)
        ax1.set_ylim(0, max(r_avgs) * 1.25)
        for bar, val in zip(bars, r_avgs):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 30,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=10)

        # annotate % drops
        bare = r_avgs[0]
        for i in range(1, 3):
            pct = (r_avgs[i] - bare) / bare * 100
            ax1.annotate(
                f"{pct:+.0f}%",
                xy=(x[i], r_avgs[i] / 2),
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold",
            )

        # right: accuracy
        bars2 = ax2.bar(x, accs, color=colors, zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(VARIANT_LABELS, fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_title("Accuracy", fontsize=12)
        ax2.set_ylim(0, 115)
        for bar, val in zip(bars2, accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


TASK_IDS = [f"R{i:02d}" for i in range(1, 11)]

TASK_LABELS = {
    "R01": "R01 Bird & trains",
    "R02": "R02 Work rate",
    "R03": "R03 Reverse %",
    "R04": "R04 Age puzzle",
    "R05": "R05 Mixture",
    "R06": "R06 Knights/knaves*",
    "R07": "R07 Sequence",
    "R08": "R08 Scheduling",
    "R09": "R09 Safe codes",
    "R10": "R10 Magic square",
}


# ── Chart 4: heatmap — avg reasoning tokens per task × model ─────────────────

def chart_task_heatmap(records: list[dict], out: Path) -> None:
    stats: dict = defaultdict(list)
    for r in records:
        if r["task_id"] in TASK_IDS:
            stats[(r["task_id"], r["model"])].append(r["reasoning_tokens"])

    matrix = np.array([
        [_avg(stats[(tid, m)]) for m in MODELS]
        for tid in TASK_IDS
    ])

    log_matrix = np.log10(np.clip(matrix, 1, None))

    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(log_matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODEL_LABELS, fontsize=11)
        ax.set_yticks(range(len(TASK_IDS)))
        ax.set_yticklabels([TASK_LABELS[t] for t in TASK_IDS], fontsize=10)

        # annotate each cell with raw value
        for i, tid in enumerate(TASK_IDS):
            for j, m in enumerate(MODELS):
                val = matrix[i, j]
                color = "white" if log_matrix[i, j] > 2.8 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("log₁₀(reasoning tokens)", fontsize=10)
        cbar.set_ticks([1, 2, 3, 4])
        cbar.set_ticklabels(["10", "100", "1K", "10K"])

        ax.set_title("Avg reasoning tokens per problem and model\n(all variants combined)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(top=False, bottom=False, left=False)

        ax.note = ax.annotate(
            "* R06 is unsolvable — no consistent knight/knave assignment exists",
            xy=(0, -0.08), xycoords="axes fraction",
            fontsize=8, color="#666",
        )

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── Chart 5: per-task politeness effect on Opus ──────────────────────────────

def chart_opus_per_task_politeness(records: list[dict], out: Path) -> None:
    rstats: dict = defaultdict(list)
    for r in records:
        if r["model"] == "claude-opus-4-8" and r["task_id"] in TASK_IDS:
            rstats[(r["task_id"], r["variant"])].append(r["reasoning_tokens"])

    pct_changes = []
    for tid in TASK_IDS:
        bare = _avg(rstats[(tid, "bare")])
        over = _avg(rstats[(tid, "overly_polite")])
        pct = (over - bare) / bare * 100 if bare else 0
        pct_changes.append((tid, pct, bare, over))

    pct_changes.sort(key=lambda x: x[1])
    tids, pcts, bares, overs = zip(*pct_changes)

    colors = ["#2980B9" if p < 0 else "#E74C3C" for p in pcts]

    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(range(len(tids)), pcts, color=colors, height=0.6, zorder=3)
        ax.axvline(0, color="#333", linewidth=1.2, zorder=4)

        ax.set_yticks(range(len(tids)))
        ax.set_yticklabels([TASK_LABELS[t] for t in tids], fontsize=10)
        ax.set_xlabel("% change in reasoning tokens (bare → overly polite)", fontsize=11)
        ax.set_title("Claude Opus 4.8 — politeness effect by problem\n(negative = fewer tokens when polite)",
                     fontsize=13, fontweight="bold", pad=12)

        ax.grid(axis="x", alpha=0.35, linestyle="--", zorder=0)

        for i, (pct, bare, over) in enumerate(zip(pcts, bares, overs)):
            pct_label = f"{pct:+.0f}%"
            val_label = f"({bare:.0f}→{over:.0f})"
            if pct < 0:
                # % inside the bar (white, right-aligned near bar end)
                ax.text(pct / 2, i, pct_label, va="center", ha="center",
                        fontsize=9, color="white", fontweight="bold")
                # raw values just right of zero line
                ax.text(0.5, i, val_label, va="center", ha="left",
                        fontsize=8, color="#555")
            else:
                # % inside bar
                ax.text(pct / 2, i, pct_label, va="center", ha="center",
                        fontsize=9, color="white", fontweight="bold")
                # raw values just past bar end
                ax.text(pct + 0.5, i, val_label, va="center", ha="left",
                        fontsize=8, color="#555")

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── Chart 6: Opus per-task grouped bars (all 3 variants) ─────────────────────

def chart_opus_per_task_variants(records: list[dict], out: Path) -> None:
    rstats: dict = defaultdict(list)
    for r in records:
        if r["model"] == "claude-opus-4-8" and r["task_id"] in TASK_IDS:
            rstats[(r["task_id"], r["variant"])].append(r["reasoning_tokens"])

    x = np.arange(len(TASK_IDS))
    width = 0.28
    offsets = [-width, 0, width]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (variant, label) in enumerate(zip(VARIANTS, VARIANT_LABELS)):
            avgs = [_avg(rstats[(tid, variant)]) for tid in TASK_IDS]
            ax.bar(x + offsets[i], avgs, width, label=label,
                   color=PALETTE[variant], zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in TASK_IDS],
                           rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Avg reasoning tokens", fontsize=11)
        ax.set_title("Claude Opus 4.8 — reasoning tokens per problem by politeness",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(framealpha=0.7, fontsize=10)

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── Chart 7: cost vs accuracy scatter — the value story ──────────────────────

def chart_cost_vs_accuracy(records: list[dict], out: Path) -> None:
    from collections import defaultdict

    costs   = defaultdict(list)
    correct = defaultdict(int)
    total   = defaultdict(int)
    rtoken  = defaultdict(list)

    for r in records:
        m = r["model"]
        costs[m].append(r.get("cost_usd", 0))
        rtoken[m].append(r["reasoning_tokens"])
        q = (r.get("quality") or {})
        if q.get("correct") is not None:
            total[m] += 1
            if q["correct"]:
                correct[m] += 1

    model_data = {
        m: {
            "cost":     sum(costs[m]) / len(costs[m]),
            "accuracy": correct[m] / total[m] * 100 if total[m] else 0,
            "tokens":   sum(rtoken[m]) / len(rtoken[m]),
        }
        for m in MODELS
    }

    dot_colors = {
        "claude-opus-4-8":   "#2C3E50",
        "claude-sonnet-4-6": "#E74C3C",
        "gpt-5.5":           "#27AE60",
    }
    label_offsets = {
        "claude-opus-4-8":   (0.001, -0.8),
        "claude-sonnet-4-6": (0.001,  0.4),
        "gpt-5.5":           (-0.005, 0.4),
    }

    with plt.rc_context({**STYLE, "axes.grid": True}):
        fig, ax = plt.subplots(figsize=(8, 6))

        for m, label in zip(MODELS, MODEL_LABELS):
            d = model_data[m]
            ax.scatter(d["cost"], d["accuracy"],
                       s=d["tokens"] / 4,          # bubble size = reasoning tokens
                       color=dot_colors[m], zorder=5, alpha=0.9,
                       edgecolors="white", linewidths=1.5)

            dx, dy = label_offsets[m]
            ax.annotate(
                f"{label}\n{d['accuracy']:.0f}% accurate\n"
                f"${d['cost']:.4f}/trial  •  {d['tokens']:.0f} reasoning tokens",
                xy=(d["cost"], d["accuracy"]),
                xytext=(d["cost"] + dx, d["accuracy"] + dy),
                fontsize=9, color=dot_colors[m], fontweight="bold",
                va="top" if dy < 0 else "bottom",
            )

        ax.set_xlabel("Avg cost per trial (USD)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(91, 102)
        ax.set_xlim(0.028, 0.072)

        ax.set_title("Cost vs accuracy by model\n(bubble size = avg reasoning tokens)",
                     fontsize=13, fontweight="bold", pad=12)

        # quadrant annotation
        ax.annotate("← cheaper  |  more accurate ↑",
                    xy=(0.5, 0.04), xycoords="axes fraction",
                    fontsize=8, color="#999", ha="center")

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graded", type=Path, default=GRADED_PATH)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = load_data(args.graded)
    print(f"Loaded {len(records)} graded trials")

    chart_reasoning_bars(records, args.out / "01_reasoning_by_model_variant.png")
    chart_opus_boxplot(records, args.out / "02_opus_reasoning_distribution.png")
    chart_reasoning_vs_accuracy(records, args.out / "03_opus_reasoning_vs_accuracy.png")
    chart_task_heatmap(records, args.out / "04_task_difficulty_heatmap.png")
    chart_opus_per_task_politeness(records, args.out / "05_opus_per_task_politeness_effect.png")
    chart_opus_per_task_variants(records, args.out / "06_opus_per_task_variants.png")
    chart_cost_vs_accuracy(records, args.out / "07_cost_vs_accuracy.png")


if __name__ == "__main__":
    main()
