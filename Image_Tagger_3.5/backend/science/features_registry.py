"""
Canonical feature registry for Image Tagger.

This module loads the forward-looking CNfA feature/attribute list from
a JSONL file produced from David's v7 and Goldilocks spreadsheets.

It is intentionally file-backed (not DB-backed) for now, so that the
Feature Navigator GUI can browse the ontology without schema churn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


FEATURES_PATH = Path(__file__).with_name("features_canonical.jsonl")


@dataclass
class FeatureDefinition:
    key: str
    category: str
    tier: str
    label: str
    status: str = "active"
    type: str = "continuous"  # binary | ordinal | categorical | continuous
    group: Optional[str] = None
    description: Optional[str] = None
    cfa_relevance: Optional[str] = None
    source: Optional[str] = None
    scale: Optional[Dict[str, Any]] = None
    methods: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDefinition":
        return cls(
            key=data.get("key", ""),
            category=data.get("category", "unknown"),
            tier=data.get("tier", "L4"),
            label=data.get("label", data.get("key", "")),
            status=data.get("status", "active"),
            type=data.get("type", "continuous"),
            group=data.get("group"),
            description=data.get("description"),
            cfa_relevance=data.get("cfa_relevance"),
            source=data.get("source"),
            scale=data.get("scale"),
            methods=data.get("methods"),
        )


@lru_cache(maxsize=1)
def load_features() -> List[FeatureDefinition]:
    feats: List[FeatureDefinition] = []
    if not FEATURES_PATH.exists():
        return feats
    import json

    with FEATURES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            feats.append(FeatureDefinition.from_dict(data))
    return feats


def list_features(
    tier: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> List[FeatureDefinition]:
    feats = load_features()
    result: List[FeatureDefinition] = []
    for feat in feats:
        if tier and feat.tier != tier:
            continue
        if category and feat.category != category:
            continue
        if status and feat.status != status:
            continue
        result.append(feat)
    return result


def get_feature(key: str) -> Optional[FeatureDefinition]:
    for feat in load_features():
        if feat.key == key:
            return feat
    return None
