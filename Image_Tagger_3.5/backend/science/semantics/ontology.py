"""
Architectural parts and semantics ontology.

This module provides a forward-looking taxonomy for architectural
elements (Tier L3). Detection and labeling may use VLMs or learned
detectors, but the ontology itself should remain stable over time.
"""

from __future__ import annotations

from typing import Dict, List


ARCH_PARTS: Dict[str, List[str]] = {
    "structural": [
        "column",
        "beam",
        "arch",
        "vault",
        "truss",
        "buttress",
    ],
    "enclosure": [
        "wall",
        "floor",
        "ceiling",
        "parapet",
        "balustrade",
    ],
    "openings": [
        "door",
        "window",
        "skylight",
        "clerestory",
        "oculus",
    ],
    "circulation": [
        "stair",
        "ramp",
        "corridor",
        "landing",
        "bridge",
    ],
    "assemblies": [
        "arcade",
        "colonnade",
        "atrium",
        "courtyard",
        "nave",
        "apse",
    ],
    "furnishing": [
        "bench",
        "altar",
        "desk",
        "chair",
        "table",
        "partition",
    ],
}

SYNONYMS: Dict[str, List[str]] = {
    "clerestory": ["high window band", "upper window strip"],
    "oculus": ["round skylight", "circular opening"],
}
