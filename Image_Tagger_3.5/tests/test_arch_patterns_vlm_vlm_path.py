"""
Basic wiring test for the ArchPatternsVLMAnalyzer VLM path.

We stub out backend.services.vlm.get_vlm_engine to return a fake engine
so this test runs without network or API keys.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.science.core import AnalysisFrame
from backend.science.semantics.arch_patterns_vlm import ArchPatternsVLMAnalyzer
from backend.services import vlm as vlm_mod


class _FakeEngine(vlm_mod.VLMEngine):
    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        # Minimal deterministic payload exercising two keys.
        return {
            "patterns": [
                {
                    "key": "arch.pattern.prospect_strong",
                    "present": 0.8,
                    "confidence": 0.9,
                    "evidence": "Clear outward view through large windows.",
                },
                {
                    "key": "arch.pattern.refuge_strong",
                    "present": 0.3,
                    "confidence": 0.7,
                    "evidence": "Limited alcove-like seating.",
                },
            ]
        }


def test_arch_patterns_vlm_wiring_uses_vlm_and_writes_attributes(monkeypatch=None) -> None:
    # Patch get_vlm_engine to return our fake engine.
    original_get = vlm_mod.get_vlm_engine

    def _fake_get(provider_override: str | None = None) -> vlm_mod.VLMEngine:  # type: ignore[override]
        return _FakeEngine()

    vlm_mod.get_vlm_engine = _fake_get  # type: ignore[assignment]

    try:
        # Construct a minimal frame with a valid RGB image.
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        frame = AnalysisFrame(image_id=1, original_image=img)

        analyzer = ArchPatternsVLMAnalyzer()
        analyzer.analyze(frame)

        # We expect numeric attributes for at least the two active keys.
        assert "arch.pattern.prospect_strong" in frame.attributes
        assert "arch.pattern.refuge_strong" in frame.attributes

        # Metadata should contain the candidates list and engine name.
        meta = frame.metadata.get("arch.patterns.candidates", {})
        assert isinstance(meta.get("candidates"), list)
        assert meta.get("engine") == "_FakeEngine"
    finally:
        # Restore original factory.
        vlm_mod.get_vlm_engine = original_get  # type: ignore[assignment]
