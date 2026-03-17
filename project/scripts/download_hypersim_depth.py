#!/usr/bin/env python3
"""
Download Hypersim GT depth (distance_from_camera) HDF5 files for the 420-image manifest.

Uses remotezip to fetch only the specific geometry HDF5 files via HTTP range requests —
no full scene zips (~2-7 GB each) are downloaded.  Each depth file is ~1-3 MB.

Depth files land at:
    data/hypersim_pilot_420/scenes/{scene_name}/images/
        scene_{camera}_geometry_hdf5/frame.{frame_id:04d}.distance_from_camera.hdf5

Usage:
    cd project/
    python scripts/download_hypersim_depth.py
    python scripts/download_hypersim_depth.py --workers 4 --skip-existing
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

BASE_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/"
    "hypersim/v1/scenes/{scene_name}.zip"
)

# Inside each scene zip (no leading "scenes/" — zips root directly at {scene_name}/):
#   {scene_name}/images/scene_{camera}_geometry_hdf5/frame.XXXX.depth_meters.hdf5
ZIP_DEPTH_TEMPLATE = (
    "{scene_name}/images/scene_{camera}_geometry_hdf5/"
    "frame.{frame_id:04d}.depth_meters.hdf5"
)

# Where to write locally (relative to data/hypersim_pilot_420/):
#   scenes/{scene_name}/images/scene_{camera}_geometry_hdf5/...
LOCAL_DEPTH_TEMPLATE = (
    "scenes/{scene_name}/images/scene_{camera}_geometry_hdf5/"
    "frame.{frame_id:04d}.depth_meters.hdf5"
)


def download_scene_depth(
    scene_name: str,
    frames: list[tuple[str, int]],
    output_dir: Path,
    skip_existing: bool,
    max_retries: int = 3,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Fetch depth HDF5 files for one scene via remotezip range requests.

    Args:
        frames: list of (camera, frame_id) tuples needed from this scene.
    Returns:
        (scene_name, [(local_rel_path, status), ...])
    """
    url = BASE_URL.format(scene_name=scene_name)
    results: list[tuple[str, str]] = []

    for attempt in range(1, max_retries + 1):
        try:
            with RemoteZip(url) as rz:
                for camera, frame_id in frames:
                    zip_path = ZIP_DEPTH_TEMPLATE.format(
                        scene_name=scene_name, camera=camera, frame_id=frame_id
                    )
                    local_rel = LOCAL_DEPTH_TEMPLATE.format(
                        scene_name=scene_name, camera=camera, frame_id=frame_id
                    )
                    dest = output_dir / local_rel
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    if skip_existing and dest.exists():
                        results.append((local_rel, "skipped"))
                        continue

                    try:
                        data = rz.read(zip_path)
                        dest.write_bytes(data)
                        results.append((local_rel, "ok"))
                    except KeyError:
                        results.append((local_rel, f"not_found_in_zip:{zip_path}"))
            break  # all frames for this scene handled — exit retry loop
        except Exception as exc:
            if attempt == max_retries:
                for camera, frame_id in frames:
                    local_rel = LOCAL_DEPTH_TEMPLATE.format(
                        scene_name=scene_name, camera=camera, frame_id=frame_id
                    )
                    results.append((local_rel, f"error:{exc}"))
            else:
                time.sleep(2 ** attempt)

    return scene_name, results


def parse_manifest(manifest_path: Path) -> dict[str, list[tuple[str, int]]]:
    """Group manifest rows by scene, returning {scene_name: [(camera, frame_id), ...]}."""
    by_scene: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with manifest_path.open() as fh:
        for row in csv.DictReader(fh):
            # file_path: scenes/{scene}/images/scene_{cam}_final_preview/frame.XXXX.tonemap.jpg
            parts = row["file_path"].split("/")
            camera = parts[3].replace("scene_", "").replace("_final_preview", "")
            frame_id = int(parts[4].replace("frame.", "").replace(".tonemap.jpg", ""))
            by_scene[row["scene_name"]].append((camera, frame_id))
    return dict(by_scene)


def main() -> int:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "configs" / "hypersim_image_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "hypersim_pilot_420",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel scene connections (default 8). Each uses HTTP range requests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally.",
    )
    args = parser.parse_args()

    by_scene = parse_manifest(args.manifest)
    total_files = sum(len(v) for v in by_scene.values())

    print(f"Manifest : {args.manifest}")
    print(f"Output   : {args.output_dir}")
    print(f"Scenes   : {len(by_scene)}")
    print(f"Depth files to fetch: {total_files}")
    print(f"Workers  : {args.workers}")
    print(f"Est. data: ~{total_files * 2:.0f} MB  ({total_files} × ~2 MB each)")
    print()

    ok = skipped = errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_scene_depth,
                scene,
                frames,
                args.output_dir,
                args.skip_existing,
            ): scene
            for scene, frames in by_scene.items()
        }
        with tqdm(total=total_files, unit="file", desc="Depth HDF5") as pbar:
            for future in as_completed(futures):
                scene_name, results = future.result()
                for local_rel, status in results:
                    pbar.update(1)
                    if status == "ok":
                        ok += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        errors += 1
                        tqdm.write(f"  WARN [{scene_name}] {status}")
                    pbar.set_postfix(ok=ok, skip=skipped, err=errors)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s  |  ok={ok}  skipped={skipped}  errors={errors}")

    if errors:
        print(f"\n{errors} error(s). Re-run with --skip-existing to retry only failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
