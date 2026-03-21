"""Canonical index catalog for the v3 science pipeline."""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, TypedDict


class BinInfo(TypedDict, total=False):
    field: str
    values: List[str]


class IndexInfo(TypedDict, total=False):
    label: str
    description: str
    type: Literal["float", "int", "str"]
    bins: Optional[BinInfo]
    tags: List[str]


INDEX_CATALOG: Dict[str, IndexInfo] = {
    "science.visual_richness": {
        "label": "Visual richness",
        "description": "Composite index combining color entropy, edge density, and texture variation.",
        "type": "float",
        "bins": {"field": "science.visual_richness_bin", "values": ["low", "mid", "high"]},
        "tags": ["composite", "candidate_bn_input"],
    },
    "science.organized_complexity": {
        "label": "Organized complexity",
        "description": "Composite index combining fractal dimension with organization ratio.",
        "type": "float",
        "bins": {"field": "science.organized_complexity_bin", "values": ["low", "mid", "high"]},
        "tags": ["composite", "candidate_bn_input"],
    },
}


def get_candidate_bn_keys() -> List[str]:
    return [k for k, info in INDEX_CATALOG.items() if "candidate_bn_input" in info.get("tags", [])]


def get_index_metadata() -> Dict[str, IndexInfo]:
    return INDEX_CATALOG
