#!/usr/bin/env python3
"""
Audit VLM variance across attributes / sources.

This script is intended to be run on top of the *flat* export of the
Validation table (or any similar CSV) in order to detect attributes
for which the Vision-Language Model (VLM) is effectively "mode
collapsed" — e.g. always emitting the same probability or bin value.

Why this matters
----------------
If a given attribute (e.g. ``cog.load.high`` or
``affect.restorative.medium``) shows:

- near-zero standard deviation across images, and
- a very narrow range of values, or
- a single value dominating almost all rows,

then we cannot trust the downstream psychological / BN analysis for
that attribute. The audit report produced here is meant to be *read*
and *acted on* by a human (e.g. Chief Scientist / Science Lead) who
can then:

- adjust prompts or model configuration,
- re-bin or re-threshold the variable,
- or temporarily drop it from science exports.

Expected input
--------------
By default the script assumes a CSV with at least the following columns:

- ``attribute_key``  — the BN / science variable name
- ``source``         — e.g. ``science_pipeline.vlm_v1`` or ``manual``
- ``value``          — numeric or numeric-coded value

You can override these column names via CLI flags if your export uses
different labels.

Example usage
-------------
    python scripts/audit_vlm_variance.py \
        --input reports/bn_validations_flat.csv \
        --out reports/vlm_variance_audit.csv \
        --source-prefix science_pipeline.vlm

The output CSV will contain one row per (attribute_key, source) pair
with summary statistics and simple "flatness" flags.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple


@dataclass
class AttributeStats:
    attribute_key: str
    source: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    range_value: float
    dominant_value: float
    dominant_fraction: float
    is_low_std: bool
    is_low_range: bool
    is_mode_collapsed: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit VLM variance across attributes / sources.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV export (e.g. bn_validations_flat.csv).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the audit CSV report.",
    )
    parser.add_argument(
        "--attribute-column",
        default="attribute_key",
        help="Column name for the attribute / variable key (default: attribute_key).",
    )
    parser.add_argument(
        "--source-column",
        default="source",
        help="Column name for the source (e.g. science_pipeline.vlm_v1).",
    )
    parser.add_argument(
        "--value-column",
        default="value",
        help="Column name for the numeric value (default: value).",
    )
    parser.add_argument(
        "--source-prefix",
        default="science_pipeline.vlm",
        help=(
            "Only rows whose source column starts with this prefix are "
            "included in the audit (default: science_pipeline.vlm)."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum number of samples required for an attribute/source pair "
             "to be included in the report (default: 30).",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.02,
        help="Standard deviation threshold below which a variable is flagged "
             "as low-variance (default: 0.02).",
    )
    parser.add_argument(
        "--range-threshold",
        type=float,
        default=0.05,
        help="Range (max-min) threshold below which a variable is flagged as "
             "low-range (default: 0.05).",
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.95,
        help="If the most frequent value accounts for >= this fraction of "
             "observations, the variable is considered 'mode collapsed' "
             "(default: 0.95).",
    )
    return parser.parse_args()


def _iter_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _coerce_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        # Treat non-numeric / empty as NaN and let the stats code ignore it.
        return float("nan")


def compute_attribute_stats(
    rows: Iterable[Dict[str, str]],
    attribute_column: str,
    source_column: str,
    value_column: str,
    source_prefix: str,
    min_samples: int,
    std_threshold: float,
    range_threshold: float,
    dominance_threshold: float,
) -> List[AttributeStats]:
    by_key_source: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for row in rows:
        attr = row.get(attribute_column)
        src = row.get(source_column)
        if not attr or not src:
            continue
        if source_prefix and not src.startswith(source_prefix):
            continue

        value = _coerce_float(row.get(value_column, ""))
        if math.isnan(value):
            continue

        by_key_source[(attr, src)].append(value)

    results: List[AttributeStats] = []

    for (attr, src), values in sorted(by_key_source.items()):
        if len(values) < min_samples:
            continue

        values_sorted = sorted(values)
        count = len(values_sorted)
        mean = float(statistics.mean(values_sorted))
        std = float(statistics.pstdev(values_sorted))  # population std
        min_value = float(values_sorted[0])
        max_value = float(values_sorted[-1])
        range_value = max_value - min_value

        # Simple mode / dominance calculation
        freq: Dict[float, int] = defaultdict(int)
        for v in values_sorted:
            freq[v] += 1
        dominant_value, dominant_count = max(freq.items(), key=lambda kv: kv[1])
        dominant_fraction = dominant_count / float(count)

        is_low_std = std <= std_threshold
        is_low_range = range_value <= range_threshold
        is_mode_collapsed = dominant_fraction >= dominance_threshold

        results.append(
            AttributeStats(
                attribute_key=attr,
                source=src,
                count=count,
                mean=mean,
                std=std,
                min_value=min_value,
                max_value=max_value,
                range_value=range_value,
                dominant_value=dominant_value,
                dominant_fraction=dominant_fraction,
                is_low_std=is_low_std,
                is_low_range=is_low_range,
                is_mode_collapsed=is_mode_collapsed,
            )
        )

    return results


def write_report(path: str, stats: List[AttributeStats]) -> None:
    fieldnames = list(asdict(stats[0]).keys()) if stats else [
        "attribute_key",
        "source",
        "count",
        "mean",
        "std",
        "min_value",
        "max_value",
        "range_value",
        "dominant_value",
        "dominant_fraction",
        "is_low_std",
        "is_low_range",
        "is_mode_collapsed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(asdict(stat))


def main() -> None:
    args = _parse_args()

    rows = _iter_rows(args.input)
    stats = compute_attribute_stats(
        rows=rows,
        attribute_column=args.attribute_column,
        source_column=args.source_column,
        value_column=args.value_column,
        source_prefix=args.source_prefix,
        min_samples=args.min_samples,
        std_threshold=args.std_threshold,
        range_threshold=args.range_threshold,
        dominance_threshold=args.dominance_threshold,
    )

    if not stats:
        print("[audit_vlm_variance] No qualifying attribute/source pairs found. "
              "Check your filters and input file.")
        return

    write_report(args.out, stats)
    print(f"[audit_vlm_variance] Wrote report with {len(stats)} rows to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
