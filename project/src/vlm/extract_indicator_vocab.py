#!/usr/bin/env python3
"""
extract_indicator_vocab.py

Reads all completed VLM annotation JSONs and compiles the indicator vocabulary:
  - Unique indicator names per affordance
  - Type and polarity distributions
  - Strength-weighted frequency counts
  - Canonical list for downstream feature engineering

Outputs:
    data/vlm_annotations/indicator_vocabulary.json
    data/vlm_annotations/indicator_vocabulary_summary.txt  (human-readable)

Usage:
    python src/vlm/extract_indicator_vocab.py
    python src/vlm/extract_indicator_vocab.py --min-count 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "vlm_annotations" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "vlm_annotations"

AFFORDANCE_IDS = ["L059", "L079", "L091", "L130", "L141"]


def load_annotations(raw_dir: Path) -> list[dict]:
    """Load all annotation JSONs from the raw output directory."""
    annotations = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
            # Skip files that failed validation entirely (no indicators key).
            if "indicators" in data:
                annotations.append(data)
        except (json.JSONDecodeError, OSError):
            print(f"[WARN] Could not read: {path}")
    return annotations


def build_vocab(annotations: list[dict], min_count: int = 1) -> dict:
    """
    Build indicator vocabulary from all annotations.

    Returns a nested dict:
        vocab[affordance_id][normalised_name] = {
            "count":        int,          # number of times indicator appeared
            "type_votes":   Counter,      # type label distribution
            "polarity_votes": Counter,    # polarity distribution
            "strength_sum": int,          # sum of strength values (weighted importance)
            "avg_strength": float,
            "example_rationales": list[str],  # up to 3 unique example rationales
        }
    """
    vocab: dict[str, dict] = {aff_id: {} for aff_id in AFFORDANCE_IDS}
    global_vocab: dict = {}  # cross-affordance

    for ann in annotations:
        aff_id     = ann.get("affordance_id", "UNKNOWN")
        indicators = ann.get("indicators", [])

        for ind in indicators:
            raw_name  = ind.get("name", "").strip().lower()
            ind_type  = ind.get("type", "object")
            polarity  = ind.get("polarity", "positive")
            strength  = ind.get("strength", 1)
            rationale = ind.get("rationale", "")

            if not raw_name:
                continue

            # Per-affordance vocab.
            bucket = vocab.setdefault(aff_id, {}).setdefault(raw_name, {
                "count":              0,
                "type_votes":         Counter(),
                "polarity_votes":     Counter(),
                "strength_sum":       0,
                "example_rationales": [],
            })
            bucket["count"]          += 1
            bucket["type_votes"][ind_type]    += 1
            bucket["polarity_votes"][polarity] += 1
            bucket["strength_sum"]   += strength
            if len(bucket["example_rationales"]) < 3 and rationale:
                if rationale not in bucket["example_rationales"]:
                    bucket["example_rationales"].append(rationale)

            # Global vocab.
            g_bucket = global_vocab.setdefault(raw_name, {
                "count":          0,
                "affordances":    set(),
                "strength_sum":   0,
            })
            g_bucket["count"]        += 1
            g_bucket["affordances"].add(aff_id)
            g_bucket["strength_sum"] += strength

    # Compute averages; apply min_count filter; resolve majority-vote type/polarity.
    result: dict[str, list] = {}
    for aff_id, aff_vocab in vocab.items():
        entries = []
        for name, b in sorted(aff_vocab.items(), key=lambda x: -x[1]["count"]):
            if b["count"] < min_count:
                continue
            avg_strength = b["strength_sum"] / b["count"]
            entries.append({
                "name":               name,
                "count":              b["count"],
                "canonical_type":     b["type_votes"].most_common(1)[0][0],
                "canonical_polarity": b["polarity_votes"].most_common(1)[0][0],
                "avg_strength":       round(avg_strength, 2),
                "strength_sum":       b["strength_sum"],
                "type_distribution":  dict(b["type_votes"]),
                "polarity_distribution": dict(b["polarity_votes"]),
                "example_rationales": b["example_rationales"],
            })
        result[aff_id] = entries

    # Global cross-affordance vocabulary (serialisable: sets → sorted lists).
    global_entries = []
    for name, b in sorted(global_vocab.items(), key=lambda x: -x[1]["count"]):
        if b["count"] < min_count:
            continue
        global_entries.append({
            "name":         name,
            "count":        b["count"],
            "affordances":  sorted(b["affordances"]),
            "avg_strength": round(b["strength_sum"] / b["count"], 2),
        })
    result["_global"] = global_entries

    return result


def write_summary(vocab: dict, out_path: Path) -> None:
    """Write a human-readable summary text file."""
    lines = ["INDICATOR VOCABULARY SUMMARY", "=" * 60, ""]

    for aff_id in AFFORDANCE_IDS:
        entries = vocab.get(aff_id, [])
        lines.append(f"{'─'*40}")
        lines.append(f"Affordance: {aff_id}  ({len(entries)} unique indicators)")
        lines.append(f"{'─'*40}")

        # Top 20 by count.
        for e in entries[:20]:
            polarity_sym = "+" if e["canonical_polarity"] == "positive" else "−"
            lines.append(
                f"  [{polarity_sym}] {e['name']:<40} "
                f"n={e['count']:>4}  "
                f"type={e['canonical_type']:<10}  "
                f"avg_strength={e['avg_strength']:.1f}"
            )
        if len(entries) > 20:
            lines.append(f"  ... and {len(entries) - 20} more")
        lines.append("")

    global_entries = vocab.get("_global", [])
    lines.append("=" * 60)
    lines.append(f"GLOBAL CROSS-AFFORDANCE VOCAB ({len(global_entries)} terms)")
    lines.append("=" * 60)
    for e in global_entries[:30]:
        affs = ", ".join(e["affordances"])
        lines.append(f"  {e['name']:<45} n={e['count']:>4}  affs=[{affs}]")

    # out_path.write_text("\n".join(lines))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract indicator vocabulary from completed VLM annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir",   type=Path, default=DEFAULT_RAW_DIR,
                        help="Directory of raw annotation JSONs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory for vocabulary files")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum occurrences for an indicator to appear in vocab")
    args = parser.parse_args()

    print(f"Loading annotations from: {args.raw_dir}")
    annotations = load_annotations(args.raw_dir)
    print(f"Loaded {len(annotations)} annotation files")

    if not annotations:
        print("[WARN] No annotations found. Run run_vlm_annotation.py first.")
        return

    vocab = build_vocab(annotations, min_count=args.min_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output.
    json_path = args.output_dir / "indicator_vocabulary.json"
    with open(json_path, "w") as fh:
        json.dump(vocab, fh, indent=2)
    print(f"Vocabulary JSON written → {json_path}")

    # Human-readable summary.
    summary_path = args.output_dir / "indicator_vocabulary_summary.txt"
    write_summary(vocab, summary_path)

    # Print quick stats.
    print("\nVocabulary sizes by affordance:")
    for aff_id in AFFORDANCE_IDS:
        entries = vocab.get(aff_id, [])
        pos = sum(1 for e in entries if e["canonical_polarity"] == "positive")
        neg = len(entries) - pos
        print(f"  {aff_id}: {len(entries):>4} terms  "
              f"({pos} positive, {neg} negative)")
    print(f"  Global: {len(vocab.get('_global', []))} unique terms across all affordances")


if __name__ == "__main__":
    main()
