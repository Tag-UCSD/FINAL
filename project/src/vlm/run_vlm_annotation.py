#!/usr/bin/env python3
"""
VLM Annotation Pipeline — run_vlm_annotation.py

Annotates 420 Hypersim images × 5 affordances = 2,100 pairs using
Qwen2-VL-7B-Instruct-AWQ as the "teacher" model.

DO NOT RUN ON THIS MACHINE — requires a GPU with ≥16 GB VRAM.
See the "Running on a GPU Machine" section of the project README for
full setup instructions.

Usage (on the target GPU machine):
    python src/vlm/run_vlm_annotation.py
    python src/vlm/run_vlm_annotation.py --affordance-ids L079 L059
    python src/vlm/run_vlm_annotation.py --dry-run
    python src/vlm/run_vlm_annotation.py --image-dir data/hypersim_pilot_420

Outputs:
    data/vlm_annotations/raw/{image_id}_{affordance_id}.json  — one file per pair
    data/vlm_annotations/checkpoint.json                      — progress tracker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from string import Template
from typing import Optional

# ── Deferred heavy imports (only loaded when not --dry-run) ───────────────────
# torch, transformers, qwen_vl_utils are imported inside main() so that
# --dry-run and --help work without a GPU environment.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR  = Path(__file__).parent / "prompts"

DEFAULT_IMAGE_DIR      = PROJECT_ROOT / "data" / "hypersim_pilot_420"
DEFAULT_MANIFEST       = PROJECT_ROOT / "configs" / "hypersim_image_manifest.csv"
DEFAULT_AFFORDANCES    = PROJECT_ROOT / "configs" / "affordance_definitions.json"
DEFAULT_SEGMENTS_DIR   = PROJECT_ROOT / "data" / "segmentation_outputs"
DEFAULT_OUTPUT_DIR     = PROJECT_ROOT / "data" / "vlm_annotations" / "raw"
DEFAULT_CHECKPOINT     = PROJECT_ROOT / "data" / "vlm_annotations" / "checkpoint.json"

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct-AWQ"

# ── Schema constants ──────────────────────────────────────────────────────────
REQUIRED_TOP_KEYS  = {"affordance_id", "affordance_name", "score", "confidence",
                      "indicators", "reasoning_summary"}
REQUIRED_IND_KEYS  = {"name", "type", "polarity", "strength", "rationale"}
VALID_TYPES        = {"object", "spatial", "surface", "lighting", "layout"}
VALID_POLARITIES   = {"positive", "negative"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID):
    """Load Qwen2-VL-7B-Instruct-AWQ with automatic device mapping."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading model: {model_id}")
    print("  (weights ~8 GB; first run will download from HuggingFace Hub)")

    # AWQ quantisation is baked into the pre-quantised checkpoint;
    # no extra quantisation config is required here.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # min/max_pixels controls how Qwen2-VL resizes input images.
    # 256*28*28 ≈ 200k px (lower bound), 1280*28*28 ≈ 1M px (upper bound).
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"  Model loaded across devices: {devices}")

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts() -> tuple[str, Template, dict]:
    """Return (system_preamble, user_Template, few_shot_by_affordance_id)."""
    system_preamble = (PROMPTS_DIR / "system_preamble.txt").read_text().strip()
    user_template   = Template((PROMPTS_DIR / "user_template.txt").read_text())

    with open(PROMPTS_DIR / "few_shot_examples.json") as fh:
        raw = json.load(fh)

    # Strip metadata key; keep affordance-keyed lists only.
    few_shot = {k: v for k, v in raw.items() if not k.startswith("_")}
    return system_preamble, user_template, few_shot


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_detected_objects(segments_path: Path) -> str:
    """Format segmentation results as a readable list for prompt injection."""
    if not segments_path.exists():
        return "(automated segmentation not yet available for this image)"

    with open(segments_path) as fh:
        segments = json.load(fh)

    objects = segments.get("detected_objects", [])
    if not objects:
        return "(no objects detected above confidence threshold)"

    # Sort by area fraction descending; cap at 20 objects to avoid token overflow.
    items = sorted(objects, key=lambda x: x.get("area_fraction", 0), reverse=True)[:20]
    parts = [
        f"{obj.get('class_name', 'unknown')} ({int(obj.get('score', 0) * 100)}%)"
        for obj in items
    ]
    return ", ".join(parts)


def format_few_shot_block(examples: list) -> str:
    """Render two few-shot examples as a text block for prompt injection."""
    lines = [
        "EXAMPLES OF EXPECTED OUTPUT FORMAT",
        "(These are text descriptions only — they are NOT the image you are assessing.)",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        label = "High" if ex["score"] >= 5 else "Low"
        lines.append(f"[Example {i} — {label} score: {ex['score']}/7]")
        lines.append(f"Scene: {ex['scene_description']}")
        lines.append(f"Expected JSON:\n{json.dumps(ex['expected_output'], indent=2)}")
        lines.append("")
    lines.append("---")
    lines.append("Now assess the ACTUAL IMAGE provided above.")
    return "\n".join(lines)


def build_messages(
    image_path: Path,
    affordance_id: str,
    affordance_def: dict,
    detected_objects_str: str,
    few_shot_examples: list,
    system_preamble: str,
    user_template: Template,
    *,
    simplified: bool = False,
) -> list[dict]:
    """Assemble the Qwen2-VL message list for one image × affordance pair."""

    few_shot_block = "" if simplified else format_few_shot_block(few_shot_examples)

    user_text = user_template.safe_substitute(
        affordance_id=affordance_id,
        affordance_name=affordance_def["name"],
        affordance_definition=affordance_def["definition"],
        detected_objects_list=detected_objects_str,
        few_shot_block=few_shot_block,
    )

    if simplified:
        # Append a terse reminder for the retry attempt.
        user_text = user_text.rstrip() + (
            "\n\nIMPORTANT: Output ONLY valid JSON matching the structure above."
            " No markdown, no explanation, no preamble."
        )

    return [
        {"role": "system", "content": system_preamble},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": user_text},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, processor, messages: list[dict],
                  max_new_tokens: int = 1024) -> str:
    """Run Qwen2-VL inference; return raw generated text."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding ≡ temperature=0.0
            repetition_penalty=1.05,  # mild penalty to avoid JSON repetition loops
        )

    # Decode only newly generated tokens (strip the echoed prompt).
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & validation
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_output(raw_text: str) -> Optional[dict]:
    """Extract JSON object from raw model output; handles markdown fences."""
    text = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end   = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    # Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract outermost {...} block.
    start = text.find("{")
    end   = text.rfind("}") + 1
    if 0 <= start < end:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def validate_output(data: dict, affordance_id: str) -> tuple[bool, list[str]]:
    """Validate a parsed annotation against the required schema.

    Returns (is_valid, list_of_error_strings).
    """
    errors: list[str] = []

    # Top-level required keys.
    missing_top = REQUIRED_TOP_KEYS - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")

    # score — integer in [1, 7].
    score = data.get("score")
    if not isinstance(score, int) or not (1 <= score <= 7):
        errors.append(f"score must be int in [1,7], got: {score!r}")

    # confidence — float in [0, 1].
    conf = data.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errors.append(f"confidence must be float in [0,1], got: {conf!r}")

    # indicators — non-empty list with valid sub-fields.
    indicators = data.get("indicators", [])
    if not isinstance(indicators, list) or len(indicators) == 0:
        errors.append("indicators must be a non-empty list")
    else:
        for i, ind in enumerate(indicators):
            missing_ind = REQUIRED_IND_KEYS - set(ind.keys())
            if missing_ind:
                errors.append(f"indicators[{i}] missing keys: {missing_ind}")
            if ind.get("type") not in VALID_TYPES:
                errors.append(
                    f"indicators[{i}].type invalid: {ind.get('type')!r} "
                    f"(must be one of {VALID_TYPES})"
                )
            if ind.get("polarity") not in VALID_POLARITIES:
                errors.append(
                    f"indicators[{i}].polarity invalid: {ind.get('polarity')!r}"
                )
            strength = ind.get("strength")
            if not isinstance(strength, int) or not (1 <= strength <= 3):
                errors.append(
                    f"indicators[{i}].strength must be int in [1,3], got: {strength!r}"
                )

    # reasoning_summary — non-empty string.
    if not isinstance(data.get("reasoning_summary"), str) or \
            not data["reasoning_summary"].strip():
        errors.append("reasoning_summary must be a non-empty string")

    # affordance_id — must match expected.
    if data.get("affordance_id") != affordance_id:
        errors.append(
            f"affordance_id mismatch: expected {affordance_id!r}, "
            f"got {data.get('affordance_id')!r}"
        )

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> set[str]:
    """Return set of completed pair keys ("image_id_affordance_id")."""
    if not path.exists():
        return set()
    with open(path) as fh:
        data = json.load(fh)
    return set(data.get("completed", []))


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {"completed": sorted(completed), "count": len(completed)},
            fh,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run VLM affordance annotation (Qwen2-VL-7B-Instruct-AWQ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-dir",    type=Path, default=DEFAULT_IMAGE_DIR,
                   help="Root dir containing Hypersim images (scenes/ sub-dir)")
    p.add_argument("--manifest",     type=Path, default=DEFAULT_MANIFEST,
                   help="Path to hypersim_image_manifest.csv")
    p.add_argument("--affordances",  type=Path, default=DEFAULT_AFFORDANCES,
                   help="Path to affordance_definitions.json")
    p.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR,
                   help="Directory with {image_id}_segments.json files (optional)")
    p.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for per-pair annotation JSONs")
    p.add_argument("--checkpoint",   type=Path, default=DEFAULT_CHECKPOINT,
                   help="Checkpoint file path")
    p.add_argument("--checkpoint-interval", type=int, default=50,
                   help="Save checkpoint every N completed pairs")
    p.add_argument("--affordance-ids", nargs="+", default=None,
                   metavar="ID",
                   help="Restrict to specific affordance IDs (e.g. L079 L059)")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Stop after N pairs (useful for smoke-testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the first 3 formatted prompts without running inference")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # ── Load configs ─────────────────────────────────────────────────────────
    import pandas as pd

    manifest = pd.read_csv(args.manifest)

    with open(args.affordances) as fh:
        raw_defs = json.load(fh)
    affordance_defs: dict = {
        k: v for k, v in raw_defs.items() if not k.startswith("_")
    }

    if args.affordance_ids:
        missing_ids = set(args.affordance_ids) - set(affordance_defs)
        if missing_ids:
            sys.exit(f"[ERROR] Unknown affordance IDs requested: {missing_ids}")
        affordance_defs = {k: v for k, v in affordance_defs.items()
                          if k in args.affordance_ids}

    print(f"Manifest      : {len(manifest)} images")
    print(f"Affordances   : {list(affordance_defs.keys())}")

    system_preamble, user_template, few_shot_by_id = load_prompts()

    # ── Build ordered work list ───────────────────────────────────────────────
    all_pairs: list[tuple[str, str, Path]] = []
    for _, row in manifest.iterrows():
        image_id = row["image_id"]
        image_path = args.image_dir / row["file_path"]
        for aff_id in affordance_defs:
            all_pairs.append((image_id, aff_id, image_path))

    if args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]

    print(f"Total pairs   : {len(all_pairs)}")

    # ── Skip completed pairs ─────────────────────────────────────────────────
    completed = load_checkpoint(args.checkpoint)
    remaining = [
        (iid, aid, ip) for iid, aid, ip in all_pairs
        if f"{iid}_{aid}" not in completed
    ]
    print(f"Completed     : {len(completed)}  |  Remaining: {len(remaining)}")

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Printing first 3 formatted prompts:\n")
        for image_id, aff_id, image_path in remaining[:3]:
            seg_path = args.segments_dir / f"{image_id}_segments.json"
            det_obj  = format_detected_objects(seg_path)
            msgs     = build_messages(
                image_path, aff_id, affordance_defs[aff_id],
                det_obj, few_shot_by_id[aff_id],
                system_preamble, user_template,
            )
            print(f"{'─'*60}")
            print(f"Pair: {image_id} × {aff_id}")
            print(f"Image path: {image_path}  (exists: {image_path.exists()})")
            for msg in msgs:
                role    = msg["role"]
                content = msg["content"]
                if isinstance(content, str):
                    snippet = content[:400]
                    print(f"\n[{role}]\n{snippet}{'...' if len(content) > 400 else ''}")
                else:
                    for part in content:
                        if part["type"] == "text":
                            snippet = part["text"][:400]
                            print(f"\n[{role}/text]\n"
                                  f"{snippet}{'...' if len(part['text']) > 400 else ''}")
                        else:
                            print(f"\n[{role}/{part['type']}] {part.get('image', '')}")
        return

    if not remaining:
        print("All pairs already completed. Exiting.")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model()

    # ── Prepare output directory ─────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Annotation loop ───────────────────────────────────────────────────────
    from tqdm import tqdm

    latencies:    list[float] = []
    errors_total: int         = 0
    retries_total: int        = 0

    pbar = tqdm(remaining, desc="Annotating", unit="pair", dynamic_ncols=True)

    for pair_idx, (image_id, aff_id, image_path) in enumerate(pbar):
        pair_key    = f"{image_id}_{aff_id}"
        output_path = args.output_dir / f"{pair_key}.json"

        # Guard: image must exist.
        if not image_path.exists():
            tqdm.write(f"[WARN]  Image not found — skipping: {image_path}")
            errors_total += 1
            continue

        seg_path         = args.segments_dir / f"{image_id}_segments.json"
        detected_objects = format_detected_objects(seg_path)
        affordance_def   = affordance_defs[aff_id]
        few_shot_exs     = few_shot_by_id.get(aff_id, [])

        t_start = time.perf_counter()
        result: Optional[dict] = None
        final_attempt          = 0

        for attempt in range(2):
            simplified = (attempt == 1)
            if simplified:
                retries_total += 1
                tqdm.write(f"[RETRY] {pair_key} — switching to simplified prompt")

            try:
                messages = build_messages(
                    image_path, aff_id, affordance_def,
                    detected_objects, few_shot_exs,
                    system_preamble, user_template,
                    simplified=simplified,
                )
                raw_output = run_inference(model, processor, messages)
                parsed     = parse_json_output(raw_output)

                if parsed is None:
                    tqdm.write(
                        f"[WARN]  {pair_key} attempt {attempt+1}: "
                        "could not parse JSON from model output"
                    )
                    continue  # try simplified prompt

                is_valid, val_errors = validate_output(parsed, aff_id)

                if not is_valid:
                    tqdm.write(
                        f"[WARN]  {pair_key} attempt {attempt+1}: "
                        f"validation errors: {val_errors}"
                    )
                    if attempt == 0:
                        continue  # retry once

                    # On second attempt, save with error flag rather than discard.
                    parsed["_validation_errors"] = val_errors

                final_attempt = attempt
                result = parsed
                break

            except Exception as exc:
                tqdm.write(f"[ERROR] {pair_key} attempt {attempt+1}: {exc}")
                errors_total += 1
                if attempt == 1:
                    break

        latency = time.perf_counter() - t_start
        latencies.append(latency)

        if result is not None:
            result["_meta"] = {
                "image_id":   image_id,
                "image_path": str(image_path),
                "model_id":   MODEL_ID,
                "latency_s":  round(latency, 2),
                "attempt":    final_attempt + 1,
            }
            with open(output_path, "w") as fh:
                json.dump(result, fh, indent=2)
            completed.add(pair_key)
        else:
            errors_total += 1
            tqdm.write(f"[FAIL]  {pair_key} — no valid output produced; skipping")

        # Checkpoint on interval.
        if (pair_idx + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, completed)
            tqdm.write(f"[CKPT]  Saved checkpoint at {len(completed)} pairs")

        # Update progress bar suffix with ETA.
        if latencies:
            avg_lat   = sum(latencies) / len(latencies)
            remaining_count = len(remaining) - (pair_idx + 1)
            eta_s     = avg_lat * remaining_count
            eta_str   = f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60):02d}m"
            pbar.set_postfix({
                "avg_s":   f"{avg_lat:.1f}",
                "ETA":     eta_str,
                "errors":  errors_total,
                "retries": retries_total,
            })

    # Final checkpoint.
    save_checkpoint(args.checkpoint, completed)

    print(
        f"\nFinished.\n"
        f"  Completed : {len(completed)}\n"
        f"  Errors    : {errors_total}\n"
        f"  Retries   : {retries_total}\n"
    )
    if latencies:
        avg = sum(latencies) / len(latencies)
        total_min = sum(latencies) / 60
        print(f"  Avg latency : {avg:.1f} s/pair")
        print(f"  Total time  : {total_min:.1f} min")


if __name__ == "__main__":
    main()
