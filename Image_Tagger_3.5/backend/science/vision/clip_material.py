"""
clip_material.py — OneFormer + SigLIP2 Material Identification Pipeline.

Two-stage pipeline:
  1. OneFormerVisualizer (from segmentation.py) → semantic-labeled instance masks
  2. SigLIP2 (google/siglip2-so400m-patch16-naflex) → zero-shot material ID per instance

Scoring:
  raw sigmoid(logits) — calibrated confidence in [0, 1].
  Ambiguous predictions are flagged as "Indeterminate Material" when the margin
  between 1st and 2nd candidate scores is below MIN_MARGIN.

After scoring, spatial consistency voting propagates the highest-confidence
material to same-class instances that scored as Indeterminate.

Usage:
    # Standalone (loads its own models)
    pipeline = MaterialIdentificationPipeline.from_pretrained()
    results  = pipeline.run(pil_image)

    # Re-use detections already computed by SegmentationAnalyzer
    pipeline = MaterialIdentificationPipeline.from_pretrained()
    results  = pipeline.run_from_frame(frame)   # reads frame.metadata

The results list is what the /materials2 debug endpoint serialises to JSON.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("v3.science.clip_material")


# ──────────────────────────────────────────────────────────────────────────────
# ADE20K Material Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

ADE20K_MATERIAL_TAXONOMY: Dict[str, Dict] = {
    "structural": {
        "ids": [0, 1, 42, 49, 56, 84, 116, 146],
        "candidates": [
            "painted wall", "plaster", "concrete", "brick",
            "stone", "marble", "tile", "wood paneling",
            "stucco", "drywall", "limestone", "glass",
        ],
    },
    "floors": {
        "ids": [3, 53, 94, 121, 136],
        "candidates": [
            "wood", "marble", "stone", "tile",
            "concrete", "carpet", "vinyl", "laminate",
            "terrazzo", "brick", "cork", "resin",
        ],
    },
    "ceilings": {
        "ids": [5, 85],
        "candidates": [
            "painted drywall", "plaster", "wood",
            "concrete", "acoustic tile", "metal",
            "glass", "fabric",
        ],
    },
    "textiles": {
        "ids": [7, 18, 19, 23, 26, 28, 30, 31, 39, 57, 81, 97, 131],
        "candidates": [
            "fabric", "leather", "velvet", "linen",
            "wool", "silk", "cotton", "suede",
            "synthetic fiber", "vinyl", "shearling", "boucle",
        ],
    },
    "millwork": {
        "ids": [10, 24, 35, 44, 45, 62, 70, 73, 99, 118],
        "candidates": [
            "wood", "painted wood", "lacquer", "laminate",
            "metal", "glass", "rattan", "wicker",
            "melamine", "bamboo", "thermofoil",
        ],
    },
    "countertops": {
        "ids": [33, 103, 159],
        "candidates": [
            "marble", "quartz", "granite", "wood",
            "concrete", "stainless steel", "porcelain",
            "stone", "corian", "glass", "slate",
        ],
    },
    "furniture": {
        "ids": [15, 19, 25, 66, 69, 132, 133, 157],
        "candidates": [
            "wood", "metal", "glass", "marble",
            "plastic", "rattan", "upholstered fabric",
            "leather", "acrylic", "stone", "concrete",
        ],
    },
    "apertures": {
        "ids": [8, 14, 38, 58, 63, 86, 95, 105],
        "candidates": [
            "glass", "wood", "metal", "aluminum",
            "steel", "vinyl", "fabric", "brass",
            "bronze", "frosted glass", "painted wood",
        ],
    },
    "lighting": {
        "ids": [36, 82, 130],
        "candidates": [
            "brass", "metal", "glass", "ceramic",
            "wood", "concrete", "fabric shade",
            "chrome", "bronze", "copper", "marble",
        ],
    },
    "decorative": {
        "ids": [34, 64, 72, 96, 102, 125, 128, 138],
        "candidates": [
            "ceramic", "glass", "metal", "marble",
            "wood", "stone", "fabric", "paper",
            "plastic", "terracotta", "brass", "bronze",
        ],
    },
    "fixtures": {
        "ids": [32, 43, 65, 78, 85, 107, 115, 148],
        "candidates": [
            "stainless steel", "porcelain", "ceramic", "chrome",
            "glass", "acrylic", "cast iron", "brass",
            "plastic", "fiberglass", "metal", "enamel",
        ],
    },
    "rugs": {
        "ids": [28],
        "candidates": [
            "wool", "cotton", "jute", "sisal",
            "synthetic fiber", "silk", "leather", "hide",
            "shearling", "woven fabric", "braided fiber",
        ],
    },
    "books_paper": {
        "ids": [67, 240],
        "candidates": [
            "paper", "cardboard", "fabric cover",
            "leather cover", "plastic cover", "linen binding",
        ],
    },
    "storage": {
        "ids": [35, 44, 45, 73],
        "candidates": [
            "wood", "painted wood", "lacquer", "laminate",
            "metal", "melamine", "thermofoil", "glass",
        ],
    },
}

_DEFAULT_CANDIDATES = [
    "wood", "metal", "glass", "ceramic", "plastic",
    "fabric", "stone", "concrete", "painted surface",
]

# Build fast lookups
_ID_TO_CANDIDATES: Dict[int, List[str]] = {}
_ID_TO_GROUP:      Dict[int, str]       = {}
for _group_name, _group in ADE20K_MATERIAL_TAXONOMY.items():
    for _cid in _group["ids"]:
        _ID_TO_CANDIDATES.setdefault(_cid, [])
        for _c in _group["candidates"]:
            if _c not in _ID_TO_CANDIDATES[_cid]:
                _ID_TO_CANDIDATES[_cid].append(_c)
        _ID_TO_GROUP[_cid] = _group_name


def get_material_candidates(class_id: int) -> List[str]:
    return _ID_TO_CANDIDATES.get(class_id, _DEFAULT_CANDIDATES)


def get_material_group(class_id: int) -> str:
    return _ID_TO_GROUP.get(class_id, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# MaterialIdentificationPipeline
# ──────────────────────────────────────────────────────────────────────────────

class MaterialIdentificationPipeline:
    """
    Two-stage pipeline: OneFormer segmentation → SigLIP2 material identification.

    Can be constructed directly (pass your own model instances) or via
    from_pretrained() which lazy-loads the default models.
    """

    MIN_MARGIN = 0.05   # gap between top-2 scores below which result is "Indeterminate"

    def __init__(self, oneformer_model, oneformer_processor, siglip_model, siglip_processor):
        import torch
        from backend.science.vision.segmentation import OneFormerVisualizer

        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.oneformer   = OneFormerVisualizer(oneformer_model, oneformer_processor)
        self.siglip      = siglip_model.to(self.device)
        self.siglip_proc = siglip_processor

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        oneformer_name: str = "shi-labs/oneformer_ade20k_swin_tiny",
        siglip_name:    str = "google/siglip2-so400m-patch16-naflex",
    ) -> "MaterialIdentificationPipeline":
        """Lazy-load both models and return a ready pipeline."""
        import torch
        from transformers import (
            OneFormerForUniversalSegmentation,
            OneFormerProcessor,
            Siglip2Model,
            Siglip2Processor,
        )

        logger.info(f"Loading OneFormer: {oneformer_name}")
        oneformer_proc  = OneFormerProcessor.from_pretrained(oneformer_name)
        oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(oneformer_name)

        logger.info(f"Loading SigLIP2: {siglip_name}")
        siglip_proc  = Siglip2Processor.from_pretrained(siglip_name)
        siglip_model = Siglip2Model.from_pretrained(siglip_name)

        return cls(oneformer_model, oneformer_proc, siglip_model, siglip_proc)

    @classmethod
    def from_frame_models(cls, frame) -> Optional["MaterialIdentificationPipeline"]:
        """
        Build a pipeline reusing the OneFormer model already loaded in
        a segmented AnalysisFrame (avoids double-loading).
        Returns None if the frame hasn't been segmented yet.
        """
        import torch
        from transformers import Siglip2Model, Siglip2Processor

        visualizer = frame.metadata.get("oneformer_visualizer")
        if visualizer is None:
            logger.warning("Frame has no cached OneFormer visualizer — run SegmentationAnalyzer first.")
            return None

        siglip_name  = "google/siglip2-so400m-patch16-naflex"
        logger.info(f"Loading SigLIP2: {siglip_name}")
        siglip_proc  = Siglip2Processor.from_pretrained(siglip_name)
        siglip_model = Siglip2Model.from_pretrained(siglip_name)

        instance = cls.__new__(cls)
        instance.device      = "cuda" if torch.cuda.is_available() else "cpu"
        instance.oneformer   = visualizer
        instance.siglip      = siglip_model.to(instance.device)
        instance.siglip_proc = siglip_proc
        return instance

    # ── SigLIP2 Scoring ────────────────────────────────────────────────────────

    def _score_materials(
        self, crop: Image.Image, class_id: int
    ) -> tuple[str, List[tuple[str, float]], float]:
        """
        Score material candidates for one instance crop.

        NAFlex requirements:
          - prompt: "this is a photo of {label}."  (lowercase)
          - max_length=64, padding="max_length"
          - max_num_patches=256

        Returns:
            group          : material group name
            ranked         : list of (material_label, score) sorted desc
            margin         : score gap between 1st and 2nd candidates
        """
        import torch

        candidates = get_material_candidates(class_id)
        group      = get_material_group(class_id)
        prompts    = [f"this is a photo of {c.lower()}." for c in candidates]

        inputs = self.siglip_proc(
            text=prompts,
            images=crop,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            max_num_patches=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.siglip(**inputs).logits_per_image[0]
            scores = torch.sigmoid(logits).cpu().numpy()

        ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        return group, ranked, margin

    # ── Spatial Consistency Voting ─────────────────────────────────────────────

    def _apply_spatial_voting(self, results: List[Dict]) -> List[Dict]:
        """
        For each semantic class, propagate the highest-confidence material to
        any instance of the same class that scored as Indeterminate.
        """
        best_per_class: Dict[str, Dict] = {}
        for r in results:
            cls = r["class_name"]
            if cls not in best_per_class or r["margin"] > best_per_class[cls]["margin"]:
                best_per_class[cls] = r

        for r in results:
            if r["top_material"] == "Indeterminate Material":
                best = best_per_class[r["class_name"]]
                if best["top_material"] != "Indeterminate Material":
                    r["top_material"] = best["top_material"] + " (voted)"
                    r["top_score"]    = best["top_score"]
                    r["vote_source"]  = best["instance_idx"]
                else:
                    r["vote_source"] = None
            else:
                r["vote_source"] = None

        return results

    # ── Reporting ──────────────────────────────────────────────────────────────

    def _print_results(self, results: List[Dict]) -> None:
        print("\n" + "=" * 70)
        print(f"MATERIAL IDENTIFICATION SUMMARY — {len(results)} instances")
        print("=" * 70)

        for r in results:
            status = ""
            if r["top_material"] == "Indeterminate Material":
                status = "  ⚠ ambiguous"
            elif r.get("vote_source"):
                status = f"  ← voted from #{r['vote_source']}"

            print(
                f"\nInstance #{r['instance_idx']:3d} — {r['class_name'].upper()} "
                f"[{r['material_group']}] (seg conf: {r['seg_confidence']:.3f}){status}"
            )
            print(
                f"  Top material: {r['top_material']:<35s} "
                f"(score: {r['top_score']:.3f}, margin: {r['margin']:.3f})"
            )
            if r["top_material"] != "Indeterminate Material":
                print("  Top 5 candidates:")
                for label, score in r["material_scores"][:5]:
                    bar = "█" * int(score * 40)
                    print(f"    {label:<35s} {score:.3f}  {bar}")

    # ── Core Run ───────────────────────────────────────────────────────────────

    def _score_crops(self, crops: List[Dict]) -> List[Dict]:
        """Score all crops and return result dicts (before voting)."""
        results = []
        for crop_data in crops:
            group, material_scores, margin = self._score_materials(
                crop_data["crop"], crop_data["class_id"]
            )
            top_label, top_score = material_scores[0]
            if margin < self.MIN_MARGIN:
                top_label = "Indeterminate Material"

            results.append(dict(
                instance_idx=crop_data["instance_idx"],
                class_id=crop_data["class_id"],
                class_name=crop_data["class_name"],
                seg_confidence=crop_data["confidence"],
                material_group=group,
                mask=crop_data["mask"],
                crop=crop_data["crop"],
                top_material=top_label,
                top_score=top_score,
                margin=margin,
                material_scores=material_scores,
                vote_source=None,
            ))
        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, image: Image.Image, show_voting_report: bool = True) -> List[Dict]:
        """
        Full pipeline: segment → score → vote → report.

        Args:
            image              : PIL Image to analyse
            show_voting_report : Whether to print voting stats

        Returns:
            List of result dicts, one per instance:
              instance_idx, class_id, class_name, seg_confidence,
              material_group, mask, crop, top_material, top_score,
              margin, material_scores, vote_source
        """
        print("\n▶ Stage 1: OneFormer segmentation...")
        detections, sem_map_np, instance_map = self.oneformer.segment(image)
        if detections is None:
            print("No instances detected.")
            return []

        print("\n▶ Stage 2: SigLIP2 material identification...")
        crops   = self.oneformer.get_instance_crops(image, detections, padding=0)
        results = self._score_crops(crops)

        print("\n▶ Stage 3: Spatial consistency voting...")
        results = self._apply_spatial_voting(results)

        if show_voting_report:
            voted = [r for r in results if r.get("vote_source")]
            indet = [r for r in results if r["top_material"] == "Indeterminate Material"]
            print(f"  {len(voted)} instances inherited material via voting")
            print(f"  {len(indet)} instances remain Indeterminate")

        self._print_results(results)
        return results

    def run_from_frame(
        self, frame, show_voting_report: bool = True
    ) -> List[Dict]:
        """
        Run only the SigLIP2 scoring stage, reusing detections already stored
        in frame.metadata by SegmentationAnalyzer.analyze().

        Skips OneFormer inference entirely — much faster when the frame has
        already been through the segmentation pipeline step.
        """
        detections = frame.metadata.get("oneformer_detections")
        if detections is None:
            logger.warning("No cached detections in frame — falling back to full run.")
            image_pil = (
                Image.fromarray(frame.original_image)
                if isinstance(frame.original_image, np.ndarray)
                else frame.original_image
            )
            return self.run(image_pil, show_voting_report=show_voting_report)

        image_pil = (
            Image.fromarray(frame.original_image)
            if isinstance(frame.original_image, np.ndarray)
            else frame.original_image
        )

        print("\n▶ Stage 2 (reusing cached detections): SigLIP2 material identification...")
        crops   = self.oneformer.get_instance_crops(image_pil, detections, padding=0)
        results = self._score_crops(crops)

        print("\n▶ Stage 3: Spatial consistency voting...")
        results = self._apply_spatial_voting(results)

        if show_voting_report:
            voted = [r for r in results if r.get("vote_source")]
            indet = [r for r in results if r["top_material"] == "Indeterminate Material"]
            print(f"  {len(voted)} instances inherited material via voting")
            print(f"  {len(indet)} instances remain Indeterminate")

        self._print_results(results)
        return results

    # ── Serialise for API ──────────────────────────────────────────────────────

    @staticmethod
    def to_json_safe(results: List[Dict]) -> List[Dict]:
        """
        Strip non-serialisable fields (numpy masks, PIL crops) for JSON responses.
        Returns a clean list ready for FastAPI to serialise.
        """
        out = []
        for r in results:
            out.append(dict(
                instance_idx=r["instance_idx"],
                class_id=r["class_id"],
                class_name=r["class_name"],
                seg_confidence=round(float(r["seg_confidence"]), 4),
                material_group=r["material_group"],
                top_material=r["top_material"],
                top_score=round(float(r["top_score"]), 4),
                margin=round(float(r["margin"]), 4),
                material_scores=[
                    {"material": label, "score": round(float(score), 4)}
                    for label, score in r["material_scores"][:10]
                ],
                vote_source=r.get("vote_source"),
            ))
        return out