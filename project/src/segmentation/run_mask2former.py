#!/usr/bin/env python3
"""Run Mask2Former panoptic segmentation over the Hypersim pilot manifest."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for GT extraction only
    h5py = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "hypersim_image_manifest.csv"
DEFAULT_TAXONOMY_MAP = PROJECT_ROOT / "configs" / "coco_to_taxonomy_map.json"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "hypersim_pilot_420"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "segmentation_outputs"
MODEL_ID = "facebook/mask2former-swin-large-coco-panoptic"
PANOPTIC_THING_ID_CUTOFF = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--taxonomy-map", type=Path, default=DEFAULT_TAXONOMY_MAP)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-ids", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(output_dir: Path, level: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_mask2former.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_manifest(manifest_path: Path, image_ids: list[str] | None, max_images: int | None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if image_ids:
        wanted = set(image_ids)
        manifest = manifest[manifest["image_id"].isin(wanted)].copy()
    if max_images is not None:
        manifest = manifest.head(max_images).copy()
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No images selected from manifest.")
    return manifest


def load_taxonomy_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as fh:
        data = json.load(fh)
    return data["classes"]


def resize_and_pad(image: Image.Image, target_size: int) -> tuple[Image.Image, dict[str, Any]]:
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)

    pad_left = (target_size - resized_w) // 2
    pad_top = (target_size - resized_h) // 2
    pad_right = target_size - resized_w - pad_left
    pad_bottom = target_size - resized_h - pad_top
    padded = ImageOps.expand(resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    meta = {
        "original_size": (height, width),
        "resized_size": (resized_h, resized_w),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "scale": scale,
        "target_size": target_size,
    }
    return padded, meta


def unpad_and_resize_panoptic_map(padded_map: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    pad_left, pad_top, pad_right, pad_bottom = meta["padding"]
    resized_h, resized_w = meta["resized_size"]
    cropped = padded_map[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    image = Image.fromarray(cropped.astype(np.int32), mode="I")
    original_h, original_w = meta["original_size"]
    restored = image.resize((original_w, original_h), resample=Image.Resampling.NEAREST)
    return np.asarray(restored, dtype=np.int32)


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, mask.astype(np.uint8), allow_pickle=False)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    return {
        "encoding": "zlib_numpy_uint8_base64",
        "shape": list(mask.shape),
        "dtype": "uint8",
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def compute_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    return [float(xs.mean() / max(width - 1, 1)), float(ys.mean() / max(height - 1, 1))]


def build_segment_records(
    panoptic_map: np.ndarray,
    segments_info: list[dict[str, Any]],
    id2label: dict[int, str],
    taxonomy_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments_for_npz: list[dict[str, Any]] = []
    segments_for_json: list[dict[str, Any]] = []
    image_area = float(panoptic_map.shape[0] * panoptic_map.shape[1])

    for segment in segments_info:
        segment_id = int(segment["id"])
        class_id = int(segment["label_id"])
        class_label = id2label[class_id]
        taxonomy_term = taxonomy_map.get(class_label, {}).get("taxonomy_term")

        mask = panoptic_map == segment_id
        if not mask.any():
            continue

        mask_payload = encode_mask(mask)
        box = compute_box(mask)
        centroid = compute_centroid(mask)
        area_fraction = float(mask.sum() / image_area)
        score = float(segment.get("score", 0.0))
        is_thing = class_id < PANOPTIC_THING_ID_CUTOFF

        record = {
            "segment_id": segment_id,
            "coco_class_id": class_id,
            "coco_class_label": class_label,
            "taxonomy_term": taxonomy_term,
            "is_thing": is_thing,
            "mask": mask_payload,
            "bounding_box": box,
            "centroid": centroid,
            "area_fraction": area_fraction,
            "confidence_score": score,
        }
        segments_for_npz.append(record)
        human_record = dict(record)
        human_record["mask"] = {
            "encoding": mask_payload["encoding"],
            "shape": mask_payload["shape"],
            "dtype": mask_payload["dtype"],
            "compressed_bytes": len(base64.b64decode(mask_payload["data"])),
            "data_base64": mask_payload["data"],
        }
        segments_for_json.append(human_record)

    return segments_for_npz, segments_for_json


def save_per_image_outputs(
    image_id: str,
    output_dir: Path,
    panoptic_map: np.ndarray,
    segments_for_npz: list[dict[str, Any]],
    segments_for_json: list[dict[str, Any]],
) -> None:
    npz_path = output_dir / f"{image_id}_panoptic.npz"
    json_path = output_dir / f"{image_id}_segments.json"

    np.savez_compressed(
        npz_path,
        panoptic_map=panoptic_map.astype(np.int32),
        segments=np.array(segments_for_npz, dtype=object),
    )

    detected_objects = [
        {
            "segment_id": seg["segment_id"],
            "class_name": seg["coco_class_label"],
            "taxonomy_term": seg["taxonomy_term"],
            "score": seg["confidence_score"],
            "area_fraction": seg["area_fraction"],
            "is_thing": seg["is_thing"],
        }
        for seg in segments_for_json
    ]
    payload = {
        "image_id": image_id,
        "num_segments": len(segments_for_json),
        "detected_objects": detected_objects,
        "segments": segments_for_json,
    }
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def maybe_extract_hypersim_gt(row: pd.Series, image_root: Path, output_dir: Path) -> dict[str, Any]:
    rel_path = Path(row["file_path"])
    frame_stem = rel_path.name.replace(".tonemap.jpg", "")
    scene_dir = image_root / rel_path.parts[0] / row["scene_name"]
    geometry_dir = scene_dir / "images" / rel_path.parent.name.replace("final_preview", "geometry_hdf5")
    semantic_path = geometry_dir / f"{frame_stem}.semantic.hdf5"
    semantic_instance_path = geometry_dir / f"{frame_stem}.semantic_instance.hdf5"

    status = {
        "image_id": row["image_id"],
        "semantic_path": str(semantic_path),
        "semantic_instance_path": str(semantic_instance_path),
        "available": False,
        "extracted": False,
        "notes": None,
    }

    if not semantic_path.exists() or not semantic_instance_path.exists():
        status["notes"] = (
            "GT segmentation not found locally. Hypersim image-level labels require the per-scene "
            "`scene_cam_XX_geometry_hdf5/frame.IIII.semantic.hdf5` and "
            "`frame.IIII.semantic_instance.hdf5` files from the official scene archives."
        )
        return status

    if h5py is None:
        status["notes"] = "GT segmentation files exist, but `h5py` is not installed."
        return status

    with h5py.File(semantic_path, "r") as fh:
        semantic = np.array(fh["dataset"])
    with h5py.File(semantic_instance_path, "r") as fh:
        semantic_instance = np.array(fh["dataset"])

    gt_out = output_dir / f"{row['image_id']}_hypersim_gt.npz"
    np.savez_compressed(
        gt_out,
        semantic=semantic.astype(np.int32),
        semantic_instance=semantic_instance.astype(np.int32),
    )
    status["available"] = True
    status["extracted"] = True
    status["notes"] = "Extracted local Hypersim GT segmentation."
    return status


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def load_model_and_processor(model_id: str, device: torch.device) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    dtype = preferred_dtype(device)
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)

    try:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=False,
        )
    except Exception:
        logging.exception("Falling back to float32 model weights after initial load failure.")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=False,
        )

    model.to(device)
    model.eval()
    return processor, model


def main() -> int:
    args = parse_args()
    log_path = setup_logging(args.output_dir, args.log_level)
    device = resolve_device(args.device)
    logging.info("Using device=%s", device)
    logging.info("Log file: %s", log_path)

    manifest = load_manifest(args.manifest, args.image_ids, args.max_images)
    taxonomy_map = load_taxonomy_map(args.taxonomy_map)
    processor, model = load_model_and_processor(args.model_id, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    summary_rows: list[dict[str, Any]] = []
    gt_status_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    failed_images: list[str] = []
    zero_detection_images: list[str] = []
    succeeded = 0

    progress = tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Mask2Former")
    for row in progress:
        row_series = pd.Series(row._asdict())
        image_id = row.image_id
        json_path = args.output_dir / f"{image_id}_segments.json"
        npz_path = args.output_dir / f"{image_id}_panoptic.npz"

        if args.skip_existing and json_path.exists() and npz_path.exists():
            logging.info("Skipping existing outputs for %s", image_id)
            continue

        image_path = args.image_root / row.file_path
        start = time.perf_counter()
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            padded_image, resize_meta = resize_and_pad(image, args.target_size)
            inputs = processor(images=padded_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            result = processor.post_process_panoptic_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                overlap_mask_area_threshold=args.overlap_mask_area_threshold,
                label_ids_to_fuse=set(),
                target_sizes=[(args.target_size, args.target_size)],
            )[0]

            padded_map = result["segmentation"].detach().cpu().numpy().astype(np.int32)
            panoptic_map = unpad_and_resize_panoptic_map(padded_map, resize_meta)
            segments_for_npz, segments_for_json = build_segment_records(
                panoptic_map=panoptic_map,
                segments_info=result["segments_info"],
                id2label=id2label,
                taxonomy_map=taxonomy_map,
            )
            save_per_image_outputs(image_id, args.output_dir, panoptic_map, segments_for_npz, segments_for_json)

            gt_status_rows.append(maybe_extract_hypersim_gt(row_series, args.image_root, args.output_dir))

            class_labels = sorted({seg["coco_class_label"] for seg in segments_for_json})
            class_counter.update(class_labels)
            elapsed = time.perf_counter() - start
            summary_rows.append(
                {
                    "image_id": image_id,
                    "num_segments": len(segments_for_json),
                    "detected_classes": class_labels,
                    "processing_time_s": elapsed,
                }
            )
            succeeded += 1
            if not segments_for_json:
                zero_detection_images.append(image_id)
                logging.warning("Zero detections for %s", image_id)
        except Exception:
            elapsed = time.perf_counter() - start
            failed_images.append(image_id)
            summary_rows.append(
                {
                    "image_id": image_id,
                    "num_segments": 0,
                    "detected_classes": [],
                    "processing_time_s": elapsed,
                }
            )
            gt_status_rows.append(
                {
                    "image_id": image_id,
                    "available": False,
                    "extracted": False,
                    "semantic_path": None,
                    "semantic_instance_path": None,
                    "notes": "Inference failed before GT extraction.",
                }
            )
            logging.exception("Failed processing image %s", image_id)
        finally:
            clear_device_cache(device)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "segmentation_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    gt_status_path = args.output_dir / "hypersim_gt_status.json"
    gt_summary = {
        "generated_at_unix_s": time.time(),
        "available_count": sum(1 for row in gt_status_rows if row["available"]),
        "extracted_count": sum(1 for row in gt_status_rows if row["extracted"]),
        "required_files": [
            "scenes/<scene_name>/images/scene_cam_XX_geometry_hdf5/frame.IIII.semantic.hdf5",
            "scenes/<scene_name>/images/scene_cam_XX_geometry_hdf5/frame.IIII.semantic_instance.hdf5",
        ],
        "source_note": (
            "These files are part of the official Hypersim per-scene archives from Apple's ml-hypersim release. "
            "The current pilot image download does not include them."
        ),
        "images": gt_status_rows,
    }
    with gt_status_path.open("w") as fh:
        json.dump(gt_summary, fh, indent=2)

    top20 = class_counter.most_common(20)
    mean_segments = float(summary_df["num_segments"].mean()) if not summary_df.empty else 0.0

    print(f"Total images processed successfully / total attempted: {succeeded} / {len(manifest)}")
    print("Distribution of detected classes (top 20 most frequent):")
    for label, count in top20:
        print(f"  {label}: {count}")
    print(f"Mean number of segments per image: {mean_segments:.2f}")
    print(f"Failed images: {failed_images if failed_images else 'None'}")
    print(f"Zero-detection images: {zero_detection_images if zero_detection_images else 'None'}")
    print(f"Segmentation summary saved to: {summary_path}")
    print(f"Hypersim GT status saved to: {gt_status_path}")
    return 0 if not failed_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
