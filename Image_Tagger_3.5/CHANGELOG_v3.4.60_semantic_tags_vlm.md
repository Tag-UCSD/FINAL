# v3.4.60 – Semantic style/room tags + cost-aware VLM

This version is strictly additive relative to v3.4.51 and focuses on
turning a subset of semantic tags from *stub* to live, cost-accounted
VLM-backed features.

## Semantic VLM analyzer

- Added `backend/science/semantics/semantic_tags_vlm.py` with a new
  `SemanticTagAnalyzer` that:
  - encodes the in-memory RGB frame as JPEG;
  - calls the configured VLM with an explicit JSON-only prompt; and
  - emits:
    - `style.*` scores (modern, traditional, minimalist, scandinavian,
      industrial, rustic, bohemian, farmhouse, japandi);
    - `spatial.room_function.*` scores (living_room, kitchen, bedroom,
      home_office, bathroom).

- The analyzer:
  - records a `semantics` metadata block with primary style and room
    function guesses;
  - runs in *metadata-only* mode when `StubEngine` is active; and
  - fails soft, recording an error flag without breaking the pipeline.

## Cost-aware semantics

- Mirrored the `CognitiveStateAnalyzer` pattern by logging a single
  cost entry per semantic VLM call via `backend.services.costs.log_vlm_usage`,
  tagged with `source="science_pipeline_semantic_tags"`.

- Kept L2 analyzers as explicit opt-ins:
  - `SciencePipelineConfig.enable_cognitive` stays `False` by default.
  - New `SciencePipelineConfig.enable_semantic` is also `False` by default.

## Registry + stub hygiene

- Removed `style.*` and `spatial.room_function.*` keys from
  `backend/science/feature_stubs.py` now that they have a concrete
  compute implementation.
- This keeps the feature coverage test strict: semantic keys are now
  treated as live features backed by the science pipeline rather than
  permanent stubs.
