#!/usr/bin/env python3
"""
Prepare a double-blind "VLM Turing Test" panel CSV.

Goal
----
Given two CSV files – one with VLM-generated tags and one with
human-generated tags – this script produces a *panel* CSV suitable
for double-blind rating by external judges.

For each (image_id, attribute_key) pair present in both inputs, we:

- Randomly assign which side is "A" and which is "B".
- Record label_A and label_B as the two competing values.
- Record which_is_vlm as "A" or "B" (for later scoring).
- Carry through any optional metadata columns requested.

The output panel can then be loaded into a simple rating UI
(Excel, Google Sheets, or a lightweight React table) where
judges rate which label is better, more plausible, or guess
which one came from the AI.

Expected input format
---------------------
The script is intentionally conservative and does not assume a
particular schema. By default it expects both CSV files to have
at least:

- image_id       (int / str)
- attribute_key  (str)
- value          (str or numeric)

You can override the column names via CLI flags.

Example
-------
    python scripts/vlm_turing_test_prep.py \
        --vlm reports/vlm_validations.csv \
        --human reports/human_validations.csv \
        --out reports/vlm_turing_panel.csv \
        --max-trials 400

"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Key = Tuple[str, str]  # (image_id, attribute_key)


@dataclass
class Record:
    image_id: str
    attribute_key: str
    value: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a VLM Turing Test panel CSV.")
    parser.add_argument("--vlm", required=True, help="CSV file with VLM outputs.")
    parser.add_argument("--human", required=True, help="CSV file with human annotations.")
    parser.add_argument("--out", required=True, help="Path to write the panel CSV.")
    parser.add_argument(
        "--image-column",
        default="image_id",
        help="Column name for the image id (default: image_id).",
    )
    parser.add_argument(
        "--attribute-column",
        default="attribute_key",
        help="Column name for the attribute key (default: attribute_key).",
    )
    parser.add_argument(
        "--value-column",
        default="value",
        help="Column name for the value / label (default: value).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=400,
        help="Maximum number of trials to include (default: 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trial sampling / A/B assignment (default: 42).",
    )
    return parser.parse_args()


def _read_records(path: Path, image_col: str, attr_col: str, value_col: str) -> Dict[Key, str]:
    records: Dict[Key, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {image_col, attr_col, value_col} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"[vlm_turing_test_prep] Missing columns in {path}: {sorted(missing)}")

        for row in reader:
            image_id = str(row[image_col])
            attr = str(row[attr_col])
            value = str(row[value_col])
            key: Key = (image_id, attr)
            # If there are multiple rows per (image, attr), last one wins;
            # this keeps the script simple and deterministic.
            records[key] = value

    return records


def build_trials(
    vlm: Dict[Key, str],
    human: Dict[Key, str],
    max_trials: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    keys = sorted(set(vlm.keys()) & set(human.keys()))
    rng.shuffle(keys)
    keys = keys[:max_trials]

    trials: List[Dict[str, str]] = []
    for idx, key in enumerate(keys, start=1):
        image_id, attr = key
        vlm_value = vlm[key]
        human_value = human[key]

        # Skip degenerate cases where both labels are identical.
        if vlm_value == human_value:
            continue

        if rng.random() < 0.5:
            label_a = vlm_value
            label_b = human_value
            which_is_vlm = "A"
        else:
            label_a = human_value
            label_b = vlm_value
            which_is_vlm = "B"

        trials.append(
            {
                "trial_id": str(idx),
                "image_id": image_id,
                "attribute_key": attr,
                "label_A": label_a,
                "label_B": label_b,
                "which_is_vlm": which_is_vlm,
                # Optional columns to be filled in by raters:
                "rater_id": "",
                "rating_A": "",
                "rating_B": "",
                "guess_is_ai": "",
                "notes": "",
            }
        )

    return trials


def write_panel(path: Path, trials: List[Dict[str, str]]) -> None:
    if not trials:
        raise SystemExit("[vlm_turing_test_prep] No non-degenerate trials constructed.")

    fieldnames = list(trials[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            writer.writerow(t)


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    vlm_records = _read_records(
        Path(args.vlm),
        image_col=args.image_column,
        attr_col=args.attribute_column,
        value_col=args.value_column,
    )
    human_records = _read_records(
        Path(args.human),
        image_col=args.image_column,
        attr_col=args.attribute_column,
        value_col=args.value_column,
    )

    trials = build_trials(vlm_records, human_records, max_trials=args.max_trials, rng=rng)
    write_panel(Path(args.out), trials)

    print(
        f"[vlm_turing_test_prep] Wrote {len(trials)} trials to {args.out}. "
        "You can now distribute this panel to human judges."
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
