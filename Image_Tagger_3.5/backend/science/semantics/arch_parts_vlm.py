"""
VLM-assisted architectural parts detection.

This analyzer is intentionally conservative: it provides *candidates*
for architectural parts with confidences and evidence strings, to be
confirmed or corrected by human annotators in the Workbench.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.core import AnalysisFrame
from backend.science.contracts import fail
from backend.science.semantics.ontology import ARCH_PARTS

# NOTE: We keep this as a stubbed integration point because the VLM
# service wiring can vary by deployment.
# STUB: integrate with backend.services.vlm.get_vlm_engine when ready.


class ArchPartsVLMAnalyzer:
    name = "arch_parts_vlm"
    tier = "L3"
    requires = ["image_url"]
    provides = ["arch.parts.candidates"]

    def __init__(self, prompt_version: str = "arch_parts_v1") -> None:
        self.prompt_version = prompt_version

    def build_prompt(self) -> str:
        categories = ", ".join(sorted(ARCH_PARTS.keys()))
        return (
            "Identify architectural elements in this interior image. "
            "Use the following high-level categories: "
            f"{categories}. "
            "For each detected element, return JSON with fields "
            "{'part': <string>, 'category': <string>, 'confidence': <0-1>, 'evidence': <short text>}."
        )

    def analyze(self, frame: AnalysisFrame) -> None:
        url = frame.metadata.get("image_url") or frame.metadata.get("image.uri")
        if not url:
            fail(frame, self.name, "no image_url in frame.metadata")
            return

        prompt = self.build_prompt()

        # STUB: call into the configured VLM engine with (url, prompt)
        # For now we just record the prompt as a placeholder and avoid emitting
        # any numeric priors that could contaminate science exports.
        candidates: List[Dict[str, Any]] = []

        frame.metadata["arch.parts.candidates"] = {
            "prompt": prompt,
            "prompt_version": self.prompt_version,
            "candidates": candidates,
            "status": "stub_not_implemented",
            "note": "VLM integration not yet wired; this is a schema stub.",
        }
