"""
Material analysis module for v3 Science Pipeline.

This module provides two levels of material detection:

1. **MaterialAnalyzer** (L0 heuristic): Fast HSV/luminance-based coverage
   estimation for wood, metal, glass, stone, plaster, and tile. No model
   download or API key required.

2. **GeminiMaterialAnalyzer** (L2 VLM): Uses Gemini Flash to identify
   materials, finishes, and textures in interior/architectural images.
   Requires a GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.
   Falls back to StubEngine when no key is configured.

Both analyzers follow the AnalysisFrame pattern and can be used
independently or together in the science pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import cv2

from backend.science.core import AnalysisFrame
from backend.services.vlm import get_vlm_engine, StubEngine

logger = logging.getLogger("v3.science.materials")

# ============================================================================
# MATERIAL TAXONOMY
# Canonical material categories for interior/architectural images
# ============================================================================

MATERIAL_CATEGORIES = [
    "wood",
    "stone",
    "marble",
    "granite",
    "concrete",
    "brick",
    "glass",
    "metal",
    "steel",
    "brass",
    "copper",
    "aluminum",
    "ceramic",
    "tile",
    "porcelain",
    "fabric",
    "leather",
    "vinyl",
    "laminate",
    "plaster",
    "drywall",
    "wallpaper",
    "carpet",
    "paint",
    "plastic",
    "rubber",
    "terrazzo",
    "travertine",
]

# Coarse grouping for summary tags
MATERIAL_GROUPS = {
    "natural_stone": ["stone", "marble", "granite", "travertine", "terrazzo"],
    "wood": ["wood", "laminate"],
    "metal": ["metal", "steel", "brass", "copper", "aluminum"],
    "glass": ["glass"],
    "ceramic": ["ceramic", "tile", "porcelain"],
    "masonry": ["concrete", "brick", "plaster", "drywall"],
    "soft": ["fabric", "leather", "carpet", "vinyl", "rubber"],
    "synthetic": ["plastic", "laminate", "vinyl", "rubber"],
    "finish": ["paint", "wallpaper"],
}

# Gemini Flash prompt for material detection
_GEMINI_MATERIAL_PROMPT = """You are an expert architectural materials analyst. Analyze this interior/architectural image and identify all visible materials, finishes, and textures.

For each material detected, provide:
- "material": the material name (e.g. "hardwood", "marble", "stainless_steel", "ceramic_tile")
- "location": where in the image (e.g. "floor", "countertop", "wall", "ceiling", "cabinet", "door", "window_frame")
- "coverage": estimated percentage of image area (0.0 to 1.0)
- "confidence": your confidence in the identification (0.0 to 1.0)
- "finish": surface finish if applicable (e.g. "matte", "glossy", "brushed", "polished", "textured", "satin")
- "color_tone": dominant color/tone (e.g. "warm oak", "cool gray", "white", "dark walnut")

Also provide:
- "dominant_material": the single most prominent material in the scene
- "material_palette": list of the top 3-5 material categories present (from: wood, stone, marble, granite, concrete, brick, glass, metal, steel, brass, copper, aluminum, ceramic, tile, porcelain, fabric, leather, vinyl, laminate, plaster, drywall, wallpaper, carpet, paint, plastic, rubber, terrazzo, travertine)
- "style_note": one sentence describing the overall material palette style (e.g. "Modern industrial with exposed concrete and steel accents")

Return STRICT JSON with this exact structure:
{
  "materials": [
    {"material": "...", "location": "...", "coverage": 0.0, "confidence": 0.0, "finish": "...", "color_tone": "..."}
  ],
  "dominant_material": "...",
  "material_palette": ["...", "..."],
  "style_note": "..."
}"""


class MaterialAnalyzer:
    """
    Heuristic-based material classification (L0).

    Uses simple HSV and luminance/texture rules to estimate coverage
    of wood, metal, and glass in the scene. Values are normalized
    coverage ratios in [0, 1].
    """

    @staticmethod
    def analyze(frame: AnalysisFrame) -> None:
        img = frame.original_image
        if img is None:
            return

        # Convert to uint8 RGB for OpenCV operations
        if img.dtype == np.float32 or img.dtype == np.float64:
            image_uint8 = (img * 255).astype(np.uint8)
        else:
            image_uint8 = img

        # Convert to HSV for material heuristics
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)

        # --- Wood heuristic (ported from v2 logic) ---
        # Brown-ish hues (approx 5–30 in OpenCV HSV),
        # with reasonable saturation and value.
        wood_mask = (
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 30)
            & (hsv[:, :, 1] > 30)
            & (hsv[:, :, 2] > 50)
        )
        wood_coverage = float(np.sum(wood_mask) / wood_mask.size)
        frame.add_attribute("material.wood_coverage", wood_coverage)

        # --- Metal heuristic (simplified from v2) ---
        # Low saturation, mid-to-high value → shiny / metallic regions.
        metal_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 150)
        metal_coverage = float(np.sum(metal_mask) / metal_mask.size)
        frame.add_attribute("material.metal_coverage", metal_coverage)

        # --- Glass heuristic (ported from v2) ---
        # High luminance + low local variance → smooth bright panes.
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        bright_mask = gray > 200

        # Local variance proxy using a smoothing kernel
        kernel = np.ones((5, 5), np.float32) / 25.0
        local_mean = cv2.filter2D(gray.astype(float), -1, kernel)
        local_var = (gray.astype(float) - local_mean) ** 2
        smooth_mask = local_var < 100.0

        glass_mask = bright_mask & smooth_mask
        glass_coverage = float(np.sum(glass_mask) / glass_mask.size)
        frame.add_attribute("material.glass_coverage", glass_coverage)
        # --- L1 material cues (read-only; support higher tiers) ---
        # Normalized brightness (0-1) from grayscale.
        gray_float = gray.astype(float) / 255.0
        brightness_mean = float(gray_float.mean())
        frame.add_attribute("materials.cues.brightness_mean", brightness_mean, confidence=0.7)

        # Global texture variance (roughness proxy).
        texture_variance = float(gray_float.var())
        frame.add_attribute("materials.cues.texture_variance", min(texture_variance * 10.0, 1.0), confidence=0.6)

        # Mean saturation and value in HSV.
        sat_mean = float(hsv[:, :, 1].mean() / 255.0)
        val_mean = float(hsv[:, :, 2].mean() / 255.0)
        frame.add_attribute("materials.cues.saturation_mean", sat_mean, confidence=0.7)
        frame.add_attribute("materials.cues.value_mean", val_mean, confidence=0.7)

        # Specularity proxy: proportion of high-value, low-saturation pixels.
        spec_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)
        specularity_proxy = float(spec_mask.sum() / spec_mask.size)
        frame.add_attribute("materials.cues.specularity_proxy", specularity_proxy, confidence=0.6)

        # --- Substrate heuristics beyond wood/metal/glass ---
        # Stone/Concrete: low saturation, mid value, higher roughness.
        stone_mask = (
            (hsv[:, :, 1] < 60) &
            (hsv[:, :, 2] > 60) &
            (hsv[:, :, 2] < 200)
        )
        stone_coverage = float(stone_mask.sum() / stone_mask.size)
        frame.add_attribute("materials.substrate.stone_concrete", stone_coverage, confidence=0.5)

        # Plaster/Gypsum: very low saturation, high value, low variance.
        plaster_mask = (
            (hsv[:, :, 1] < 30) &
            (hsv[:, :, 2] > 180)
        )
        plaster_coverage = float(plaster_mask.sum() / plaster_mask.size)
        frame.add_attribute("materials.substrate.plaster_gypsum", plaster_coverage, confidence=0.5)

        # Tile/Ceramic: bright and moderately saturated with elevated local variance.
        # We reuse local_var from the glass heuristic as a crude texture cue.
        tile_mask = (
            (hsv[:, :, 2] > 150) &
            (hsv[:, :, 1] > 40) &
            (local_var > 50.0)
        )
        tile_coverage = float(tile_mask.sum() / tile_mask.size)
        frame.add_attribute("materials.substrate.tile_ceramic", tile_coverage, confidence=0.4)


# ============================================================================
# GEMINI FLASH MATERIAL ANALYZER (L2 VLM)
# ============================================================================

class GeminiMaterialAnalyzer:
    """
    VLM-based material detection using Gemini Flash.

    Uses the configured VLM engine (defaults to Gemini Flash when
    GEMINI_API_KEY is set) to identify materials, finishes, and textures
    in interior/architectural images.

    Output:
    - Frame attributes: material.vlm.dominant, material.vlm.material_count, etc.
    - Frame metadata: material_detection (full result), material_tags (tag list)
    - Tags: "material:wood (85%)", "material_finish:matte hardwood (floor)", etc.

    Falls back gracefully to StubEngine when no API key is configured.
    """

    @staticmethod
    def analyze(
        frame: AnalysisFrame,
        provider_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run Gemini Flash material detection on the image.

        Args:
            frame: AnalysisFrame containing the image to analyze
            provider_override: Force a specific VLM provider (e.g. "gemini", "openai")

        Returns:
            Dictionary containing materials, tags, and style analysis.
            Returns a stub result if no VLM provider is configured.
        """
        img = frame.original_image
        if img is None:
            logger.warning("No image in frame, skipping material VLM analysis")
            return {"stub": True, "error": "no_image"}

        # Encode image as JPEG bytes for VLM
        if img.dtype == np.float32 or img.dtype == np.float64:
            image_uint8 = (img * 255).astype(np.uint8)
        else:
            image_uint8 = img

        # Convert RGB to BGR for OpenCV encoding
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            logger.error("Failed to encode image for VLM material analysis")
            return {"stub": True, "error": "encode_failed"}

        image_bytes = buffer.tobytes()

        # Get VLM engine
        engine = get_vlm_engine(provider_override=provider_override)

        # Check for stub mode
        if isinstance(engine, StubEngine):
            logger.info("VLM in stub mode - no material VLM analysis available")
            stub_result = {
                "stub": True,
                "engine": "StubEngine",
                "note": "Configure GEMINI_API_KEY to enable Gemini Flash material detection.",
            }
            frame.metadata["material_detection"] = stub_result
            frame.metadata["material_tags"] = []
            return stub_result

        # Call VLM
        try:
            logger.info(f"Running material detection via {type(engine).__name__}...")
            raw_result = engine.analyze_image(image_bytes, _GEMINI_MATERIAL_PROMPT)
        except Exception as e:
            logger.error(f"VLM material detection failed: {e}")
            error_result = {
                "stub": True,
                "error": str(e),
                "engine": type(engine).__name__,
            }
            frame.metadata["material_detection"] = error_result
            frame.metadata["material_tags"] = []
            return error_result

        # Parse and structure the result
        result = GeminiMaterialAnalyzer._parse_vlm_result(raw_result)

        # Generate tags from the parsed result
        tags = GeminiMaterialAnalyzer._generate_tags(result)

        # Store in frame attributes
        materials_list = result.get("materials", [])
        dominant = result.get("dominant_material", "unknown")
        palette = result.get("material_palette", [])

        frame.add_attribute("material.vlm.dominant", hash(dominant) % 1000 / 1000.0)
        frame.add_attribute("material.vlm.material_count", float(len(materials_list)))
        frame.add_attribute("material.vlm.palette_size", float(len(palette)))

        # Store coverage for top materials
        for mat_entry in materials_list[:5]:
            mat_name = mat_entry.get("material", "unknown").replace(" ", "_").lower()
            coverage = float(mat_entry.get("coverage", 0.0))
            confidence = float(mat_entry.get("confidence", 0.0))
            frame.add_attribute(
                f"material.vlm.{mat_name}_coverage",
                coverage,
                confidence=confidence,
            )

        # Store full result and tags in metadata
        result["tags"] = tags
        result["engine"] = type(engine).__name__
        frame.metadata["material_detection"] = result
        frame.metadata["material_tags"] = tags

        logger.info(
            f"Material detection: dominant={dominant}, "
            f"materials={len(materials_list)}, tags={len(tags)}"
        )

        return result

    @staticmethod
    def _parse_vlm_result(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the VLM response into a structured result."""
        result = {
            "materials": [],
            "dominant_material": "unknown",
            "material_palette": [],
            "style_note": "",
        }

        if not isinstance(raw, dict):
            logger.warning(f"VLM returned non-dict type: {type(raw)}")
            return result

        # Extract materials list
        materials = raw.get("materials", [])
        if isinstance(materials, list):
            parsed_materials = []
            for mat in materials:
                if isinstance(mat, dict):
                    parsed_materials.append({
                        "material": str(mat.get("material", "unknown")),
                        "location": str(mat.get("location", "unknown")),
                        "coverage": min(1.0, max(0.0, float(mat.get("coverage", 0.0)))),
                        "confidence": min(1.0, max(0.0, float(mat.get("confidence", 0.0)))),
                        "finish": str(mat.get("finish", "")),
                        "color_tone": str(mat.get("color_tone", "")),
                    })
            # Sort by coverage descending
            parsed_materials.sort(key=lambda x: x["coverage"], reverse=True)
            result["materials"] = parsed_materials

        # Extract dominant material
        dominant = raw.get("dominant_material", "unknown")
        if isinstance(dominant, str) and dominant:
            result["dominant_material"] = dominant

        # Extract palette
        palette = raw.get("material_palette", [])
        if isinstance(palette, list):
            result["material_palette"] = [str(p) for p in palette if isinstance(p, str)]

        # Extract style note
        style_note = raw.get("style_note", "")
        if isinstance(style_note, str):
            result["style_note"] = style_note

        return result

    @staticmethod
    def _generate_tags(result: Dict[str, Any]) -> List[str]:
        """Generate human-readable tags from the parsed material detection result."""
        tags = []

        # Dominant material tag
        dominant = result.get("dominant_material", "unknown")
        if dominant and dominant != "unknown":
            tags.append(f"material:{dominant}")

        # Individual material tags with confidence
        for mat in result.get("materials", [])[:6]:
            material_name = mat.get("material", "unknown")
            confidence = mat.get("confidence", 0.0)
            location = mat.get("location", "")
            finish = mat.get("finish", "")
            conf_pct = int(confidence * 100)

            if conf_pct >= 50:
                # High confidence: include location
                tag = f"material:{material_name} ({conf_pct}%)"
                if tag not in tags:
                    tags.append(tag)

                # Add finish tag if available
                if finish and finish not in ("", "unknown", "None"):
                    finish_tag = f"finish:{finish} {material_name} ({location})"
                    tags.append(finish_tag)

        # Palette tags
        for palette_mat in result.get("material_palette", [])[:5]:
            palette_tag = f"palette:{palette_mat}"
            if palette_tag not in tags:
                tags.append(palette_tag)

        # Style note as a tag
        style_note = result.get("style_note", "")
        if style_note and len(style_note) < 80:
            tags.append(f"material_style:{style_note}")

        return tags

    @staticmethod
    def get_tags(frame: AnalysisFrame) -> List[str]:
        """
        Get the material detection tags from a frame that has been analyzed.

        Args:
            frame: AnalysisFrame that has been processed by analyze()

        Returns:
            List of tag strings
        """
        return frame.metadata.get("material_tags", [])

    @staticmethod
    def update_image_tags(
        db_session,
        image_id: int,
        tags: List[str],
        replace_material_tags: bool = True,
    ) -> bool:
        """
        Update an image's meta_data tags with material detection results.

        Args:
            db_session: SQLAlchemy database session
            image_id: ID of the image to update
            tags: List of material tags to add
            replace_material_tags: If True, removes existing material:* tags first

        Returns:
            True if successful, False otherwise
        """
        from backend.models.assets import Image

        try:
            image = db_session.query(Image).filter(Image.id == image_id).first()
            if not image:
                logger.warning(f"Image {image_id} not found for tag update")
                return False

            # Get existing meta_data or create new
            meta_data = image.meta_data or {}
            existing_tags = meta_data.get("tags", [])

            if replace_material_tags:
                # Remove existing material tags
                existing_tags = [
                    t for t in existing_tags
                    if not t.startswith("material:")
                    and not t.startswith("finish:")
                    and not t.startswith("palette:")
                    and not t.startswith("material_style:")
                ]

            # Add new material tags
            existing_tags.extend(tags)

            # Update meta_data
            meta_data["tags"] = existing_tags
            image.meta_data = meta_data

            db_session.commit()
            logger.info(f"Updated image {image_id} with material tags: {tags}")
            return True

        except Exception as e:
            logger.error(f"Failed to update image {image_id} tags: {e}")
            db_session.rollback()
            return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_materials_vlm(
    image: np.ndarray,
    provider_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to detect materials from a single image using Gemini Flash.

    Args:
        image: RGB numpy array (H, W, 3)
        provider_override: Force a specific VLM provider

    Returns:
        Dictionary with material detection results including 'tags' key
    """
    frame = AnalysisFrame(image_id=-1, original_image=image)
    return GeminiMaterialAnalyzer.analyze(frame, provider_override=provider_override)
