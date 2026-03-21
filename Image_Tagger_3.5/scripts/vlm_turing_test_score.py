#!/usr/bin/env python3
"""
Score results from a VLM Turing Test panel.

This script consumes a CSV produced by ``vlm_turing_test_prep.py`` after
it has been annotated by human judges. Each row should correspond to
one rater's judgment for a given trial and include at least:

- trial_id        (str / int)
- which_is_vlm    ("A" or "B")
- guess_is_ai     ("A" or "B")
- rating_A        (numeric, optional)
- rating_B        (numeric, optional)
- rater_id        (str, optional)

The script computes:

- overall accuracy of guesses (<= 50% ~ indistinguishable from chance),
- per-rater accuracy,
- mean ratings for AI vs human labels where available.

The goal is not to enforce a particular statistical test but to provide
a quick sanity-check summary for the Science Lead.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class TrialStats:
    trial_id: str
    n_raters: int
    n_correct: int
    mean_rating_ai: float
    mean_rating_human: float


@dataclass
class GlobalStats:
    total_judgments: int
    total_correct: int
    overall_accuracy: float
    per_rater_accuracy: Dict[str, float]
    mean_rating_ai: float
    mean_rating_human: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score VLM Turing Test results.")
    parser.add_argument("--panel", required=True, help="CSV exported after judges have filled in the panel.")
    return parser.parse_args()


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def score_panel(path: Path) -> GlobalStats:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"trial_id", "which_is_vlm", "guess_is_ai"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"[vlm_turing_test_score] Missing required columns: {sorted(missing)}")

        per_rater_counts: Dict[str, int] = Counter()
        per_rater_correct: Dict[str, int] = Counter()

        ai_ratings: List[float] = []
        human_ratings: List[float] = []

        total_judgments = 0
        total_correct = 0

        for row in reader:
            total_judgments += 1
            which_is_vlm = row["which_is_vlm"].strip()
            guess = row["guess_is_ai"].strip()
            rater_id = row.get("rater_id", "").strip() or "UNKNOWN"

            if guess and guess in {"A", "B"} and which_is_vlm in {"A", "B"}:
                is_correct = int(guess == which_is_vlm)
                total_correct += is_correct
                per_rater_counts[rater_id] += 1
                per_rater_correct[rater_id] += is_correct

            # Ratings (optional)
            rating_a = _safe_float(row.get("rating_A", ""))
            rating_b = _safe_float(row.get("rating_B", ""))

            if which_is_vlm == "A":
                if not math.isnan(rating_a):
                    ai_ratings.append(rating_a)
                if not math.isnan(rating_b):
                    human_ratings.append(rating_b)
            elif which_is_vlm == "B":
                if not math.isnan(rating_b):
                    ai_ratings.append(rating_b)
                if not math.isnan(rating_a):
                    human_ratings.append(rating_a)

        overall_accuracy = (total_correct / total_judgments) if total_judgments else 0.0

        per_rater_accuracy = {
            rater: (per_rater_correct[rater] / count) if count else 0.0
            for rater, count in per_rater_counts.items()
        }

        def _mean(xs: List[float]) -> float:
            xs = [x for x in xs if not math.isnan(x)]
            return sum(xs) / len(xs) if xs else float("nan")

        mean_ai = _mean(ai_ratings)
        mean_human = _mean(human_ratings)

        return GlobalStats(
            total_judgments=total_judgments,
            total_correct=total_correct,
            overall_accuracy=overall_accuracy,
            per_rater_accuracy=per_rater_accuracy,
            mean_rating_ai=mean_ai,
            mean_rating_human=mean_human,
        )


def main() -> None:
    args = _parse_args()
    stats = score_panel(Path(args.panel))

    print("[vlm_turing_test_score] Global summary")
    print(f"  total_judgments : {stats.total_judgments}")
    print(f"  total_correct   : {stats.total_correct}")
    print(f"  overall_accuracy: {stats.overall_accuracy:.3f}")
    print(f"  mean_rating_ai  : {stats.mean_rating_ai:.3f}")
    print(f"  mean_rating_human: {stats.mean_rating_human:.3f}")
    print("  per_rater_accuracy:")
    for rater, acc in sorted(stats.per_rater_accuracy.items()):
        print(f"    - {rater}: {acc:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
