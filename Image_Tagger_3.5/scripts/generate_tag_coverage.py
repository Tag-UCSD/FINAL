#!/usr/bin/env python3
"""Generate a science tag coverage snapshot for Image Tagger.

This script inspects the science code and stub list to produce a
machine-readable coverage map:

  - which feature keys are known (from registry, stubs, or analyzers);
  - which ones are computed by which analyzers (via `add_attribute`);
  - which ones are marked as stubs only;
  - a coarse-grained source_type classification.

It writes:

  - `science_tag_coverage_v1.json` at the repo root; and
  - `docs/SCIENCE_TAG_MAP.md` with a human-readable summary.

Run from the repo root:

    python scripts/generate_tag_coverage.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
SCIENCE_DIR = REPO_ROOT / "backend" / "science"

# Ensure `backend` is importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These imports are intentionally late-bound for robustness.
from backend.science import feature_stubs
try:
    from backend.science.features_registry import load_features
except Exception:  # pragma: no cover - defensive
    load_features = None  # type: ignore[assignment]


@dataclass
class FeatureCoverage:
    key: str
    analyzers: List[str]
    stub: bool
    source_type: str  # math_or_deterministic | vlm_cognitive | vlm_semantic | stub_only | unassigned


class _AttributeCollector(ast.NodeVisitor):
    """Collect `frame.add_attribute("key", ...)` calls."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.current_class: str | None = None
        self.mapping: DefaultDict[str, Set[str]] = defaultdict(set)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        is_add = False
        if isinstance(func, ast.Attribute) and func.attr == "add_attribute":
            is_add = True
        elif isinstance(func, ast.Name) and func.id == "add_attribute":
            is_add = True

        if is_add and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                key = arg0.value
                owner = self.module_name
                if self.current_class:
                    owner = f"{owner}.{self.current_class}"
                self.mapping[key].add(owner)

        self.generic_visit(node)


def _collect_semantic_keys() -> Set[str]:
    """Extract canonical semantic feature keys from SemanticTagAnalyzer.

    We do not rely on `add_attribute` calls here because the analyzer uses
    an intermediate mapping (style_map / room_map) with variables.
    Instead we scan for literal strings starting with the canonical
    prefixes.
    """
    sem_file = SCIENCE_DIR / "semantics" / "semantic_tags_vlm.py"
    if not sem_file.exists():
        return set()

    try:
        source = sem_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()

    try:
        tree = ast.parse(source, filename=str(sem_file))
    except SyntaxError:
        return set()

    semantic_keys: Set[str] = set()

    class _ConstVisitor(ast.NodeVisitor):
        def visit_Constant(self, node):  # type: ignore[override]
            if isinstance(node.value, str):
                if node.value.startswith("style.") or node.value.startswith("spatial.room_function."):
                    semantic_keys.add(node.value)
            self.generic_visit(node)

    _ConstVisitor().visit(tree)
    return semantic_keys


def _collect_computed_keys() -> Dict[str, Set[str]]:
    """Scan backend/science for add_attribute calls.

    Returns a mapping: feature_key -> set of "module[.Class]" strings.
    """
    mapping: DefaultDict[str, Set[str]] = defaultdict(set)

    for path in SCIENCE_DIR.rglob("*.py"):
        # Skip internal / guard scripts that are not science analyzers.
        if path.name.endswith("_guard.py"):
            continue

        rel = path.relative_to(REPO_ROOT).with_suffix("")
        module_name = ".".join(rel.parts)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            # Guards / legacy snapshots may not parse cleanly; skip.
            continue

        collector = _AttributeCollector(module_name)
        collector.visit(tree)

        for key, owners in collector.mapping.items():
            mapping[key].update(owners)

    return dict(mapping)


def _classify_source_type(key: str, analyzers: Sequence[str], stub_keys: Set[str]) -> str:
    """Coarse classification of source type for a feature key."""
    if analyzers:
        if any("semantic_tags_vlm.SemanticTagAnalyzer" in a for a in analyzers):
            return "vlm_semantic"
        if any("context.cognitive.CognitiveStateAnalyzer" in a for a in analyzers):
            return "vlm_cognitive"
        return "math_or_deterministic"

    # No analyzers found.
    if key in stub_keys:
        return "stub_only"
    return "unassigned"


def main() -> None:
    # 1) Collect stubs and computed keys.
    stub_keys: Set[str] = set(feature_stubs.STUB_FEATURE_KEYS)
    computed = _collect_computed_keys()

    # 1b) Inject semantic keys as computed by SemanticTagAnalyzer explicitly.
    semantic_keys = _collect_semantic_keys()
    if semantic_keys:
        owner = "backend.science.semantics.semantic_tags_vlm.SemanticTagAnalyzer"
        for key in semantic_keys:
            computed.setdefault(key, set()).add(owner)

    # 2) Advisory registry from load_features(), if available and non-empty.
    registry_meta: Dict[str, dict] = {}
    registry_keys: Set[str] = set()
    feats = []
    if load_features is not None:
        try:
            feats = load_features()
        except Exception:
            feats = []

    if feats:
        for feat in feats:
            key = getattr(feat, "key", None)
            if not isinstance(key, str) or not key:
                continue
            registry_keys.add(key)
            registry_meta[key] = {
                "key": feat.key,
                "category": getattr(feat, "category", None),
                "tier": getattr(feat, "tier", None),
                "status": getattr(feat, "status", None),
                "type": getattr(feat, "type", None),
                "group": getattr(feat, "group", None),
            }
    else:
        # Fallback: take the union of stub keys and computed keys.
        registry_keys.update(stub_keys)
        registry_keys.update(computed.keys())
        for key in registry_keys:
            registry_meta[key] = {"key": key}

    coverage: Dict[str, FeatureCoverage] = {}
    counts: DefaultDict[str, int] = defaultdict(int)

    for key in sorted(registry_keys):
        analyzers = sorted(computed.get(key, []))
        source_type = _classify_source_type(key, analyzers, stub_keys)
        stub = key in stub_keys

        cov = FeatureCoverage(
            key=key,
            analyzers=analyzers,
            stub=stub,
            source_type=source_type,
        )
        coverage[key] = cov
        counts[source_type] += 1
        if stub:
            counts["stub_total"] += 1

    meta = {
        "version": 1,
        "registry_count": len(registry_keys),
        "computed_count": sum(1 for c in coverage.values() if c.analyzers),
        "stub_count": len(stub_keys),
        "counts_by_source_type": dict(counts),
    }

    # JSON snapshot
    json_payload = {
        "meta": meta,
        "coverage": {
            key: {
                "key": cov.key,
                "analyzers": cov.analyzers,
                "stub": cov.stub,
                "source_type": cov.source_type,
            }
            for key, cov in sorted(coverage.items(), key=lambda kv: kv[0])
        },
    }

    json_path = REPO_ROOT / "science_tag_coverage_v1.json"
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown summary
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "SCIENCE_TAG_MAP.md"

    lines = []
    lines.append("# Science Tag Coverage Map (v1)")
    lines.append("")
    lines.append("Autogenerated by `scripts/generate_tag_coverage.py`.")
    lines.append("")
    lines.append(f"- Total known feature keys: {meta['registry_count']}")
    lines.append(f"- Keys with at least one compute implementation: {meta['computed_count']}")
    lines.append(f"- Stub-allowed keys: {meta['stub_count']}")
    lines.append("")
    lines.append("## Breakdown by source_type")
    lines.append("")
    lines.append("| source_type | count |")
    lines.append("|------------|-------|")
    for stype, count in sorted(meta["counts_by_source_type"].items()):
        lines.append(f"| {stype} | {count} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `math_or_deterministic`: numeric features computed by the L0/L1 engines")
    lines.append("  (e.g., color histograms, texture, fractals, depth/spatial metrics).")
    lines.append("- `vlm_cognitive`: high-level cognitive/affective dimensions estimated by")
    lines.append("  the CognitiveStateAnalyzer VLM.")
    lines.append("- `vlm_semantic`: semantic tags such as style.* and spatial.room_function.*")
    lines.append("  estimated by the SemanticTagAnalyzer VLM.")
    lines.append("- `stub_only`: keys that are intentionally present in the registry but do")
    lines.append("  not yet have a compute implementation; they are tracked by")
    lines.append("  `backend/science/feature_stubs.py`.")
    lines.append("- `unassigned`: keys that are present in the union of registry/stub/computed")
    lines.append("  keys but that currently have no detectable compute implementation and are")
    lines.append("  not listed as stubs. In a healthy repository this count should be zero.")
    lines.append("  New builds should fail if it drifts above zero.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[generate_tag_coverage] Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"[generate_tag_coverage] Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"[generate_tag_coverage] Meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
