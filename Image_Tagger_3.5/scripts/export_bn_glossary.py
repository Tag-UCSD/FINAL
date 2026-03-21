"""Export BN candidate input glossary.

This script reads INDEX_CATALOG and exports a machine-readable glossary of
entries tagged as "candidate_bn_input" to docs/BN_GLOSSARY_AUTO.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.science.index_catalog import get_candidate_bn_keys, get_index_metadata


def main() -> int:
    catalog = get_index_metadata()
    keys = get_candidate_bn_keys()
    out: dict[str, dict] = {}

    for key in keys:
        info = catalog.get(key, {})
        out[key] = {
            "label": info.get("label"),
            "description": info.get("description"),
            "type": info.get("type"),
            "bins": info.get("bins"),
            "tags": list(info.get("tags", [])),
        }

    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    target = docs / "BN_GLOSSARY_AUTO.json"
    target.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[export_bn_glossary] wrote {len(out)} entries to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
