#!/usr/bin/env python3
"""
Selective Hypersim image downloader.

Uses remotezip to extract only the specific tone-mapped JPEG frames listed in
hypersim_image_manifest.csv without downloading full scene zip files (~2-7 GB each).
Each zip's central directory is fetched (~0.4s), then only the needed file bytes
are downloaded (~50-200 KB per frame instead of 2+ GB per scene).

Usage:
    python download_manifest_images.py [--manifest PATH] [--output_dir PATH] [--workers N]
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene_name}.zip"

def download_scene(scene_name, frames, output_dir, max_retries=3):
    """Download specific frames from a remote scene zip via range requests."""
    url = BASE_URL.format(scene_name=scene_name)
    results = []

    for attempt in range(1, max_retries + 1):
        try:
            with RemoteZip(url) as rz:
                for (camera, frame_id, dest_path) in frames:
                    zip_path = f"{scene_name}/images/scene_{camera}_final_preview/frame.{frame_id:04d}.tonemap.jpg"
                    dest = Path(output_dir) / dest_path
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    if dest.exists():
                        results.append((dest_path, "skipped"))
                        continue

                    try:
                        data = rz.read(zip_path)
                        dest.write_bytes(data)
                        results.append((dest_path, "ok"))
                    except KeyError:
                        results.append((dest_path, f"not_found_in_zip:{zip_path}"))
            break  # success
        except Exception as e:
            if attempt == max_retries:
                for (_, frame_id, dest_path) in frames:
                    results.append((dest_path, f"error:{e}"))
            else:
                time.sleep(2 ** attempt)

    return scene_name, results


def main():
    parser = argparse.ArgumentParser(description="Download Hypersim manifest images via remotezip.")
    parser.add_argument("--manifest", default="configs/hypersim_image_manifest.csv")
    parser.add_argument("--output_dir", default="data/hypersim_pilot_420")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel scene downloads (each uses HTTP range requests; default 8)")
    args = parser.parse_args()

    # Resolve paths relative to project root (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    manifest_path = project_root / args.manifest
    output_dir = project_root / args.output_dir

    print(f"Manifest : {manifest_path}")
    print(f"Output   : {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group manifest rows by scene
    by_scene = defaultdict(list)
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            scene = row["scene_name"]
            # file_path: scenes/{scene_name}/images/scene_{cam}_final_preview/frame.XXXX.tonemap.jpg
            parts = row["file_path"].split("/")
            camera = parts[3].replace("scene_", "").replace("_final_preview", "")
            frame_id = int(parts[4].replace("frame.", "").replace(".tonemap.jpg", ""))
            dest_path = row["file_path"]  # preserve relative path structure under output_dir
            by_scene[scene].append((camera, frame_id, dest_path))

    scenes = list(by_scene.items())
    total_frames = sum(len(v) for v in by_scene.values())
    print(f"\nScenes   : {len(scenes)}")
    print(f"Frames   : {total_frames}")
    print(f"Workers  : {args.workers}")
    print()

    ok = skipped = errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_scene, scene, frames, output_dir): scene
                   for scene, frames in scenes}
        with tqdm(total=total_frames, unit="img", desc="Downloading") as pbar:
            for future in as_completed(futures):
                scene_name, results = future.result()
                for dest_path, status in results:
                    pbar.update(1)
                    if status == "ok":
                        ok += 1
                    elif status == "skipped":
                        skipped += 1
                        pbar.set_postfix(ok=ok, skip=skipped, err=errors)
                    else:
                        errors += 1
                        tqdm.write(f"  WARN [{scene_name}] {dest_path}: {status}")
                    pbar.set_postfix(ok=ok, skip=skipped, err=errors)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s  |  ok={ok}  skipped={skipped}  errors={errors}")

    if errors:
        print(f"\n{errors} error(s) — re-run to retry failed downloads.")
        sys.exit(1)


if __name__ == "__main__":
    main()
