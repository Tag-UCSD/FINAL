"""
Feature Navigator API (v1).

Read-only API for browsing the CNfA feature/attribute ontology.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.science.features_registry import FeatureDefinition, list_features, get_feature

router = APIRouter(prefix="/v1/features", tags=["features"])


class FeatureRead(BaseModel):
    key: str
    category: str
    tier: str
    label: str
    status: str
    type: str
    group: Optional[str] = None
    description: Optional[str] = None
    cfa_relevance: Optional[str] = None
    source: Optional[str] = None
    scale: Optional[dict] = None
    methods: Optional[list] = None


@router.get("/", response_model=List[FeatureRead])
def list_all_features(
    tier: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = "active",
):
    feats = list_features(tier=tier, category=category, status=status)
    return [
        FeatureRead(
            key=f.key, category=f.category, tier=f.tier, label=f.label,
            status=f.status, type=f.type, group=f.group, description=f.description,
            cfa_relevance=f.cfa_relevance, source=f.source, scale=f.scale, methods=f.methods,
        )
        for f in feats
    ]


@router.get("/{key}", response_model=FeatureRead)
def get_feature_detail(key: str):
    feat = get_feature(key)
    if feat is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return FeatureRead(
        key=feat.key, category=feat.category, tier=feat.tier, label=feat.label,
        status=feat.status, type=feat.type, group=feat.group, description=feat.description,
        cfa_relevance=feat.cfa_relevance, source=feat.source, scale=feat.scale, methods=feat.methods,
    )
