"""
Instance and Semantic Segmentation Analyzer using OneFormer.

Architecture:
- OneFormerVisualizer: core inference class — runs semantic + panoptic passes,
  merges them into clean instance masks, generates overlays, and prints summaries.
  This is also what the CLIP material pipeline consumes directly.
- SegmentationAnalyzer: thin pipeline adapter that calls OneFormerVisualizer,
  writes metrics into AnalysisFrame, and caches the visualizer + detections in
  frame.metadata for downstream reuse (e.g. clip_material.py).

Debug API endpoints call SegmentationAnalyzer.get_segmentation_overlay() which
delegates to the appropriate overlay method on OneFormerVisualizer.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.science.core import AnalysisFrame

# Lazy-loaded globals
ONEFORMER_MODEL     = None
ONEFORMER_PROCESSOR = None
_VISUALIZER: Optional["OneFormerVisualizer"] = None

logger = logging.getLogger("v3.science.segmentation")


# =============================================================================
# OneFormerVisualizer
# =============================================================================

class OneFormerVisualizer:
    """
    Semantic-panoptic segmentation producing merged instance masks for downstream use.

    Merging strategy:
      1. Run semantic segmentation  → dense class map (H×W)
      2. Run panoptic segmentation  → object proposals with scores
      3. For each semantic class:
           a. Accept panoptic proposals whose pixels agree with the semantic
              map at >= SEM_AGREE_THRESHOLD.
           b. Mark any unclaimed semantic pixels as their own instance.
      4. Return sv.Detections + a dense instance_map (H×W int32).
    """

    MIN_PIXELS          = 2000
    SLIVER_RATIO        = 0.10
    SEM_AGREE_THRESHOLD = 0.60
    PAN_THRESHOLD       = 0.5
    PAN_OVERLAP         = 0.8

    def __init__(self, model, processor):
        self.model      = model
        self.processor  = processor
        self.id2label   = model.config.id2label
        self.model_name = model.config._name_or_path.split("/")[-1]
        self.device     = next(model.parameters()).device

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_semantic(self, image, img_size):
        import torch
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img_size]
        )[0]

    def _run_panoptic(self, image, img_size):
        import torch
        inputs = self.processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[img_size],
            threshold=self.PAN_THRESHOLD,
            overlap_mask_area_threshold=self.PAN_OVERLAP,
        )[0]

    # ── Conversion ─────────────────────────────────────────────────────────────

    def _semantic_map_np(self, semantic_map):
        return semantic_map.cpu().numpy()

    def _panoptic_masks(self, panoptic_result):
        seg_map   = panoptic_result["segmentation"].cpu().numpy()
        instances = []
        for seg in panoptic_result["segments_info"]:
            mask = seg_map == seg["id"]
            if mask.sum() >= self.MIN_PIXELS:
                instances.append((mask, seg["label_id"], seg.get("score", 1.0)))
        return instances

    # ── Merge ──────────────────────────────────────────────────────────────────

    def _panoptic_agrees_with_semantic(self, pan_mask, sem_map_np, sem_class_id):
        pan_pixels      = sem_map_np[pan_mask]
        matching_pixels = (pan_pixels == sem_class_id).sum()
        return matching_pixels / pan_pixels.size

    def _merge(self, sem_map_np, pan_instances):
        try:
            import supervision as sv
        except ImportError:
            logger.warning("supervision not installed; merge step skipped.")
            return None, None

        merged_masks, merged_class_ids, merged_confidences = [], [], []
        claimed_pixels = np.zeros(sem_map_np.shape, dtype=bool)

        for sem_class_id in np.unique(sem_map_np):
            sem_mask = sem_map_np == sem_class_id

            for pan_mask, _, pan_score in pan_instances:
                agreement = self._panoptic_agrees_with_semantic(pan_mask, sem_map_np, sem_class_id)
                if agreement < self.SEM_AGREE_THRESHOLD:
                    continue
                intersection = sem_mask & pan_mask
                if intersection.sum() < self.MIN_PIXELS:
                    continue
                if intersection.sum() / pan_mask.sum() < self.SLIVER_RATIO:
                    continue
                merged_masks.append(intersection)
                merged_class_ids.append(int(sem_class_id))
                merged_confidences.append(float(pan_score))
                claimed_pixels |= intersection

            unclaimed = sem_mask & ~claimed_pixels
            if unclaimed.sum() >= self.MIN_PIXELS:
                merged_masks.append(unclaimed)
                merged_class_ids.append(int(sem_class_id))
                merged_confidences.append(1.0)

        if not merged_masks:
            return None, None

        instance_map = np.zeros(sem_map_np.shape, dtype=np.int32)
        for idx, mask in enumerate(merged_masks):
            instance_map[mask] = idx + 1

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=np.array(merged_masks)),
            mask=np.array(merged_masks),
            class_id=np.array(merged_class_ids),
            confidence=np.array(merged_confidences),
        )
        return detections, instance_map

    # ── Color / Canvas ─────────────────────────────────────────────────────────

    def _build_class_colors(self, detections):
        try:
            import supervision as sv
        except ImportError:
            return {}
        palette        = sv.ColorPalette.DEFAULT
        unique_classes = list(dict.fromkeys(detections.class_id.tolist()))
        return {cid: palette.by_idx(i).as_rgb() for i, cid in enumerate(unique_classes)}

    def _build_instance_canvas(self, shape, detections, class_colors, contour_thickness=1):
        import cv2
        canvas = np.zeros((*shape[:2], 3), dtype=np.uint8)
        for mask, class_id in zip(detections.mask, detections.class_id):
            canvas[mask] = class_colors.get(int(class_id), (128, 128, 128))
        for mask in detections.mask:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (255, 255, 255), thickness=contour_thickness)
        return canvas

    # ── Visualization (notebook / script use) ──────────────────────────────────

    def _visualize(self, image, detections, instance_map):
        """4-panel matplotlib figure: original / semantic masks / labeled overlay / instance map."""
        import matplotlib.pyplot as plt

        try:
            import supervision as sv
        except ImportError:
            logger.warning("supervision not installed; skipping visualisation.")
            return

        fig, axes = plt.subplots(1, 4, figsize=(26, 7))
        fig.suptitle(self.model_name, fontsize=18, fontweight="bold", y=0.98)

        if detections is None or len(detections) == 0:
            for ax in axes:
                ax.imshow(image); ax.axis("off")
            axes[1].set_title("No Detections", fontsize=14, fontweight="bold")
            plt.tight_layout(); plt.show()
            return

        img_np       = np.array(image)
        class_colors = self._build_class_colors(detections)

        # Panel 0 — original
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Panel 1 — flat semantic masks
        sem_canvas = sv.MaskAnnotator(opacity=1.0, color_lookup=sv.ColorLookup.CLASS).annotate(
            scene=np.zeros_like(img_np), detections=detections
        )
        axes[1].imshow(sem_canvas)
        axes[1].set_title("Semantic Class Masks", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Panel 2 — labeled overlay (panoptic)
        overlay = sv.MaskAnnotator(opacity=0.4, color_lookup=sv.ColorLookup.CLASS).annotate(
            scene=img_np.copy(), detections=detections
        )
        labels = [
            f"{self.id2label.get(cid, f'class_{cid}')} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        overlay = sv.LabelAnnotator(
            text_position=sv.Position.CENTER, text_thickness=2,
            text_scale=0.5, text_padding=5,
        ).annotate(scene=overlay, detections=detections, labels=labels)
        axes[2].imshow(overlay)
        axes[2].set_title("Labeled Overlay", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        # Panel 3 — instance map with index labels
        inst_canvas = self._build_instance_canvas(img_np.shape, detections, class_colors)
        axes[3].imshow(inst_canvas)
        for idx, (mask, class_id) in enumerate(zip(detections.mask, detections.class_id)):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            axes[3].text(
                int(xs.mean()), int(ys.mean()),
                f"#{idx + 1}\n{self.id2label.get(int(class_id), f'class_{class_id}')}",
                fontsize=6, color="white", ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4, linewidth=0),
            )
        axes[3].set_title(
            f"Instance Map ({len(detections)} instances)", fontsize=14, fontweight="bold"
        )
        axes[3].axis("off")
        plt.tight_layout(); plt.show()

    # ── Summary ────────────────────────────────────────────────────────────────

    def _print_summary(self, detections, seg_map_np):
        if detections is None:
            print("No segments detected"); return

        total_pixels = seg_map_np.size
        class_data   = defaultdict(list)

        for idx, (mask, class_id, conf) in enumerate(
            zip(detections.mask, detections.class_id, detections.confidence)
        ):
            class_data[self.id2label.get(class_id, f"class_{class_id}")].append(
                dict(instance_idx=idx + 1, percentage=(mask.sum() / total_pixels) * 100,
                     pixels=mask.sum(), score=conf)
            )

        print("\n" + "=" * 70)
        print(f"SEMANTIC-PANOPTIC SUMMARY — {len(detections)} instances, {len(class_data)} unique classes")
        print("=" * 70)
        for class_name in sorted(class_data, key=lambda n: -sum(i["pixels"] for i in class_data[n])):
            instances  = class_data[class_name]
            total_area = sum(i["percentage"] for i in instances)
            avg_score  = sum(i["score"] for i in instances) / len(instances)
            print(f"\n{class_name.upper()}: {len(instances)} instance(s), "
                  f"{total_area:.2f}% total area, Avg Score: {avg_score:.3f}")
            for inst in instances:
                print(f"  Instance #{inst['instance_idx']:3d}: {inst['percentage']:5.2f}% "
                      f"({inst['pixels']:,} px) — Score: {inst['score']:.3f}")

    # ── Overlay helpers  (called by the debug API) ─────────────────────────────

    def build_semantic_overlay_np(
        self, image_np: np.ndarray, detections, alpha: float = 0.5
    ) -> np.ndarray:
        """Flat coloured semantic mask overlay (no labels). Returns RGB ndarray."""
        try:
            import supervision as sv
        except ImportError:
            return image_np
        if detections is None or len(detections) == 0:
            return image_np
        return sv.MaskAnnotator(opacity=alpha, color_lookup=sv.ColorLookup.CLASS).annotate(
            scene=image_np.copy(), detections=detections
        )

    def build_panoptic_overlay_np(
        self,
        image_np: np.ndarray,
        detections,
        alpha: float = 0.4,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Labelled panoptic overlay (class name + confidence). Returns RGB ndarray."""
        try:
            import supervision as sv
        except ImportError:
            return image_np
        if detections is None or len(detections) == 0:
            return image_np

        overlay = sv.MaskAnnotator(opacity=alpha, color_lookup=sv.ColorLookup.CLASS).annotate(
            scene=image_np.copy(), detections=detections
        )
        if show_labels:
            labels = [
                f"{self.id2label.get(int(cid), f'class_{cid}')} {conf:.0%}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            overlay = sv.LabelAnnotator(
                text_position=sv.Position.CENTER, text_thickness=1,
                text_scale=0.45, text_padding=4,
            ).annotate(scene=overlay, detections=detections, labels=labels)
        return overlay

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, image):
        """
        Run the full semantic → panoptic → merge pipeline.

        Returns:
            detections   : sv.Detections (merged instances, ready for SigLIP)
            sem_map_np   : np.ndarray H×W int32 semantic class map
            instance_map : np.ndarray H×W int32 instance index map
        """
        img_size      = image.size[::-1]
        sem_map_np    = self._semantic_map_np(self._run_semantic(image, img_size))
        pan_instances = self._panoptic_masks(self._run_panoptic(image, img_size))
        detections, instance_map = self._merge(sem_map_np, pan_instances)
        self._visualize(image, detections, instance_map)
        self._print_summary(detections, sem_map_np)
        return detections, sem_map_np, instance_map

    def get_instance_crops(self, image, detections, padding: int = 0) -> List[Dict]:
        """Return list of cropped instance dicts (used by clip_material pipeline)."""
        img_np = np.array(image)
        H, W   = img_np.shape[:2]
        crops  = []
        for i, (xyxy, mask, class_id, conf) in enumerate(
            zip(detections.xyxy, detections.mask, detections.class_id, detections.confidence)
        ):
            x1, y1, x2, y2 = xyxy.astype(int)
            x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
            x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
            masked        = img_np.copy()
            masked[~mask] = 0
            crops.append(dict(
                instance_idx=i + 1,
                crop=Image.fromarray(masked[y1p:y2p, x1p:x2p]),
                class_id=int(class_id),
                class_name=self.id2label.get(int(class_id), f"class_{class_id}"),
                confidence=float(conf),
                bbox=(x1p, y1p, x2p, y2p),
                mask=mask,
            ))
        return crops


# =============================================================================
# SegmentationAnalyzer  (pipeline adapter)
# =============================================================================

class SegmentationAnalyzer:
    """
    Thin pipeline adapter — calls OneFormerVisualizer, stores metrics in
    AnalysisFrame, and caches the visualizer + detections in frame.metadata
    for downstream reuse (clip_material.py).
    """

    @staticmethod
    def load_model() -> "OneFormerVisualizer":
        """Lazy-load OneFormer and return the shared OneFormerVisualizer singleton."""
        global ONEFORMER_MODEL, ONEFORMER_PROCESSOR, _VISUALIZER
        if _VISUALIZER is None:
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
            model_name = "shi-labs/oneformer_ade20k_swin_tiny"
            logger.info(f"Loading OneFormer model: {model_name}")
            ONEFORMER_PROCESSOR = OneFormerProcessor.from_pretrained(model_name)
            ONEFORMER_MODEL     = OneFormerForUniversalSegmentation.from_pretrained(model_name)
            _VISUALIZER = OneFormerVisualizer(ONEFORMER_MODEL, ONEFORMER_PROCESSOR)
            logger.info("OneFormer loaded successfully")
        return _VISUALIZER

    @staticmethod
    def analyze(
        frame: AnalysisFrame,
        use_semantic: bool = True,
        use_panoptic: bool = True,
    ) -> Dict[str, Any]:
        """
        Run OneFormer segmentation, write metrics to frame, cache in frame.metadata.

        frame.metadata keys set:
          oneformer_detections   → sv.Detections (merged)
          oneformer_sem_map_np   → np.ndarray semantic map
          oneformer_instance_map → np.ndarray instance index map
          oneformer_visualizer   → OneFormerVisualizer instance (for overlays)
        """
        visualizer = SegmentationAnalyzer.load_model()

        image_pil = (
            Image.fromarray(frame.original_image)
            if isinstance(frame.original_image, np.ndarray)
            else frame.original_image
        )
        id2label = visualizer.id2label

        logger.info("Running OneFormer segmentation (semantic + panoptic)...")
        detections, sem_map_np, instance_map = visualizer.segment(image_pil)

        results: Dict[str, Any] = {}

        # Semantic coverage from dense map
        if use_semantic and sem_map_np is not None:
            total_px = sem_map_np.size
            for cls_id in np.unique(sem_map_np):
                px = int((sem_map_np == cls_id).sum())
                if px < 100:
                    continue
                safe = id2label.get(int(cls_id), f"class_{cls_id}").replace(" ", "_")
                frame.add_attribute(f"segmentation.semantic_{safe}_coverage", px / total_px)
            results["sem_map_np"] = sem_map_np

        # Instance metrics from merged detections
        if use_panoptic and detections is not None:
            total_px   = detections.mask[0].size if len(detections.mask) else 1
            counts: Dict[str, int]   = {}
            coverages: Dict[str, float] = {}
            for mask, cls_id in zip(detections.mask, detections.class_id):
                cls_name = id2label.get(int(cls_id), f"class_{cls_id}")
                px       = int(mask.sum())
                if px < 100:
                    continue
                counts[cls_name]    = counts.get(cls_name, 0) + 1
                coverages[cls_name] = coverages.get(cls_name, 0.0) + px / total_px

            for cls_name, cnt in counts.items():
                frame.add_attribute(f"segmentation.{cls_name.replace(' ', '_')}_count", cnt)
            for cls_name, cov in coverages.items():
                frame.add_attribute(f"segmentation.{cls_name.replace(' ', '_')}_coverage", cov)

            combined      = np.zeros(sem_map_np.shape, dtype=np.uint8)
            for mask in detections.mask:
                combined |= mask.astype(np.uint8)
            scene_coverage  = combined.sum() / combined.size
            total_instances = sum(counts.values())

            frame.add_attribute("segmentation.total_objects",  total_instances)
            frame.add_attribute("segmentation.scene_coverage", scene_coverage)

            results.update(dict(
                instance_counts=counts, instance_coverages=coverages,
                total_instances=total_instances, scene_coverage=scene_coverage,
            ))

        # Cache everything for downstream use
        results["detections"]   = detections
        results["instance_map"] = instance_map
        frame.metadata.update(dict(
            oneformer_detections=detections,
            oneformer_sem_map_np=sem_map_np,
            oneformer_instance_map=instance_map,
            oneformer_visualizer=visualizer,
        ))

        logger.info(
            f"Segmentation complete: {results.get('total_instances', 0)} instances, "
            f"{results.get('scene_coverage', 0.0):.1%} scene coverage"
        )
        return results

    @staticmethod
    def get_segmentation_overlay(
        frame: AnalysisFrame,
        alpha: float = 0.5,
        overlay_type: str = "panoptic",
    ) -> Optional[np.ndarray]:
        """
        Generate an overlay PNG from detections stored in frame.metadata.

        overlay_type:
          "semantic" → flat class-coloured masks
          "panoptic" → labelled instance overlay (default)
        """
        detections = frame.metadata.get("oneformer_detections")
        visualizer = frame.metadata.get("oneformer_visualizer")
        if detections is None or visualizer is None:
            return None

        image_np = (
            frame.original_image
            if isinstance(frame.original_image, np.ndarray)
            else np.array(frame.original_image)
        )

        if overlay_type == "semantic":
            return visualizer.build_semantic_overlay_np(image_np, detections, alpha=alpha)
        return visualizer.build_panoptic_overlay_np(image_np, detections, alpha=alpha)


# Convenience wrapper
def run_segmentation_on_image(image: np.ndarray, image_id: int = -1) -> Dict[str, Any]:
    frame = AnalysisFrame(image_id=image_id, original_image=image)
    return SegmentationAnalyzer.analyze(frame)