"""Export a BN-ready dataset from a running Image Tagger v3 API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import requests


def fetch_bn_rows(api_base: str) -> Iterable[dict]:
    url = api_base.rstrip("/") + "/v1/export/bn-snapshot"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Expected a list of BNRow objects from /v1/export/bn-snapshot")
    return data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export BN-ready dataset from Image Tagger v3 API.")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--out", default="bn_dataset.jsonl")
    args = parser.parse_args()

    rows = list(fetch_bn_rows(args.api_base))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} BN rows to {out_path}")


if __name__ == "__main__":
    main()
