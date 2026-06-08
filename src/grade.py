import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

CORPUS_PATH = Path("corpus/tasks_reasoning.json")
RESULTS_PATH = Path("results/raw_reasoning.jsonl")
GRADED_PATH = Path("results/graded_reasoning.jsonl")

# Ground truth answers for R01-R10
GROUND_TRUTHS: dict[str, dict] = {
    "R01": {
        "answer": "480 miles",
        "detail": (
            "Trains are 480 miles apart at combined speed 120 mph, so they meet in 4 hours. "
            "Bird flies 120 mph for 4 hours = 480 miles total."
        ),
    },
    "R02": {
        "answer": "24/11 days (approximately 2.18 days)",
        "detail": (
            "Together for 2 days: 2*(1/8+1/12) = 5/12 done. Remaining 7/12 finished by Alice+Carol in 1 day. "
            "Alice contributes 1/8, so Carol does 7/12-1/8 = 11/24 per day → Carol alone takes 24/11 days."
        ),
    },
    "R03": {
        "answer": "approximately $69.58 wholesale cost",
        "detail": (
            "Reverse the chain: 60.75+5=65.75 (before coupon), /0.9=73.06 (before loyalty), "
            "/0.75=97.41 (before sale), /1.4=69.58 (wholesale). "
            "Exact: 65.75/0.945 ≈ $69.58."
        ),
    },
    "R04": {
        "answer": "Son is 10, father is 40 now; father will be exactly twice son's age in 20 years",
        "detail": (
            "Let son=s, father=4s. In 5 years: 4s+5=3(s+5) → s=10, father=40. "
            "Twice as old: 40+t=2(10+t) → t=20."
        ),
    },
    "R05": {
        "answer": (
            "Add 1000/3 ≈ 333.33 ml of 70% solution; "
            "total volume ≈ 533.33 ml; discard ≈ 233.33 ml to get exactly 300 ml of 55%"
        ),
        "detail": (
            "0.3(200)+0.7x=0.55(200+x) → 0.15x=50 → x=1000/3≈333.33 ml. "
            "Total=533.33 ml. Discard 533.33-300=233.33 ml."
        ),
    },
    "R06": {
        "answer": "No consistent solution exists — the four statements are logically contradictory",
        "detail": (
            "All 16 possible knight/knave assignments lead to at least one contradiction. "
            "The puzzle has no valid solution. A response recognising the contradiction is correct."
        ),
    },
    "R07": {
        "answer": (
            "Pattern: a(n) = 3×2^(n-1) − 1. "
            "Next three terms: 95, 191, 383. "
            "Smallest n exceeding 10,000 is n=13 (a(13)=12,287). "
            "Sum of first 10 terms = 3,059."
        ),
        "detail": "Each term = 2×previous + 1. Formula: a(n) = 3×2^(n-1) − 1.",
    },
    "R08": {
        "answer": (
            "6 valid orderings: "
            "(E,A,B,C,D), (E,B,A,C,D), (A,E,B,C,D), (B,E,A,C,D), (A,B,C,D,E), (B,A,C,D,E)"
        ),
        "detail": (
            "CD in positions (4,5) gives 4 orderings; CD in positions (3,4) gives 2 orderings. "
            "CD in (1,2) or (2,3) gives 0."
        ),
    },
    "R09": {
        "answer": "8 valid codes: 2871, 2961, 4932, 4752, 6813, 6723, 6543, 8514",
        "detail": (
            "First/last digit pairs (2,1),(4,2),(6,3),(8,4) yield 2+2+3+1=8 codes. "
            "All satisfy: all digits different, no zero, first=2×last, digit2>digit3, sum=18."
        ),
    },
    "R10": {
        "answer": (
            "Two valid magic squares (both acceptable): "
            "[2,9,4 / 7,5,3 / 6,1,8] or [2,7,6 / 9,5,1 / 4,3,8]"
        ),
        "detail": (
            "Magic constant=15, center=5. Given top-left=2, two solutions exist. "
            "Either is correct; extra credit for finding both."
        ),
    },
}

GRADING_PROMPT = """\
You are a math and logic grader. Determine whether a response contains the correct final answer.

PROBLEM:
{problem}

CORRECT ANSWER:
{correct_answer}

STUDENT'S RESPONSE:
{response}

Does the response arrive at the correct final answer (or a mathematically equivalent one)?
Be lenient about notation and presentation — only check whether the core answer is right.
For multi-part problems, the response is CORRECT if it gets the primary answer right even if it \
misses secondary parts.

Respond with exactly one word on the first line: CORRECT or INCORRECT
Then on the next line, one sentence explaining your judgment."""


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, dict]:
    with path.open() as f:
        return {t["task_id"]: t for t in json.load(f)}


def load_graded(path: Path = GRADED_PATH) -> set[str]:
    if not path.exists():
        return set()
    graded = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                graded.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return graded


def grade_trial(
    problem_content: str,
    task_id: str,
    response_text: str,
    client: anthropic.Anthropic,
) -> dict:
    if task_id not in GROUND_TRUTHS:
        return {"correct": None, "explanation": f"No ground truth for {task_id}"}

    truth = GROUND_TRUTHS[task_id]
    prompt = GRADING_PROMPT.format(
        problem=problem_content,
        correct_answer=f"{truth['answer']}\n\nNote: {truth['detail']}",
        response=response_text[:4000],
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    lines = text.split("\n", 1)
    verdict = lines[0].strip().upper()
    explanation = lines[1].strip() if len(lines) > 1 else ""

    return {
        "correct": verdict == "CORRECT",
        "explanation": explanation,
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=GRADED_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    already_graded = load_graded(args.output)
    client = anthropic.Anthropic()

    records = []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    pending = [r for r in records if r["trial_id"] not in already_graded]
    print(f"Trials: {len(records)}  Already graded: {len(already_graded)}  Pending: {len(pending)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    correct_count = 0

    with args.output.open("a") as out:
        for record in tqdm(pending, desc="Grading"):
            task_id = record["task_id"]
            if task_id not in corpus:
                continue
            try:
                quality = grade_trial(
                    corpus[task_id]["content"],
                    task_id,
                    record["response_text"],
                    client,
                )
            except Exception as e:
                print(f"\nERROR {record['trial_id']}: {e}")
                errors += 1
                continue

            out.write(json.dumps({**record, "quality": quality}) + "\n")
            if quality["correct"]:
                correct_count += 1

    total_graded = len(pending) - errors
    print(f"\nDone. Graded: {total_graded}  Correct: {correct_count}  Errors: {errors}")
    if total_graded > 0:
        print(f"Overall accuracy: {correct_count / total_graded * 100:.1f}%")


if __name__ == "__main__":
    main()
