"""
Contracts and helper utilities for science analyzers.

This module defines a minimal Analyzer protocol and helper methods to keep
the growing family of analyzers consistent and forward-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, List

from .core import AnalysisFrame


class Analyzer(Protocol):
    """Minimal protocol every analyzer should follow."""

    name: str
    tier: str  # e.g. "L0", "L1", "L2", "L3", "L4", "L5"
    requires: List[str]
    provides: List[str]

    def analyze(self, frame: AnalysisFrame) -> None:
        ...


@dataclass
class AnalysisError:
    analyzer: str
    reason: str
    detail: Optional[Dict[str, Any]] = None


def safe_get(frame: AnalysisFrame, key: str) -> Any:
    """Small wrapper in case we later move away from direct attributes dict."""
    return frame.attributes.get(key)


def safe_set(frame: AnalysisFrame, key: str, value: Any, confidence: float = 1.0, provenance: Optional[Dict[str, Any]] = None) -> None:
    frame.add_attribute(key, value, confidence=confidence)
    if provenance is not None:
        meta = frame.metadata.get(key, {})
        meta.update(provenance)
        frame.metadata[key] = meta


def fail(frame: AnalysisFrame, analyzer_name: str, reason: str) -> None:
    """Record a failure as metadata without poisoning numeric priors."""
    key = f"science_error.{analyzer_name}"
    frame.metadata[key] = {"reason": reason}


class AnalyzerRegistry:
    """Forward-looking registry for analyzers and their tiers/dependencies."""

    def __init__(self) -> None:
        self._analyzers: Dict[str, Analyzer] = {}

    def register(self, analyzer: Analyzer) -> None:
        self._analyzers[analyzer.name] = analyzer

    def get_all(self) -> Dict[str, Analyzer]:
        return dict(self._analyzers)

