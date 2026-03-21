# Image Tagger v3.4.32 — Restorativeness H1 heuristic in Tag Inspector

This version builds on v3.4.31_bn_canon_sanity and introduces the first explicit
"restorativeness" rule family (H1) in the Tag Inspector.

- Tag Inspector:
  * Added `_build_restorativeness_heuristic_node` to `backend/api/v1_supervision.py`.
  * The helper:
    - reads CNfA fluency + biophilia features from the `features` list returned by
      `/v1/monitor/image/{image_id}/inspector` (if present), focusing on:
        - `cnfa.biophilic.natural_material_ratio`
        - `cnfa.fluency.visual_entropy_spatial`
        - `cnfa.fluency.clutter_density_count`
        - `cnfa.fluency.processing_load_proxy`
    - computes a simple, explicitly heuristic restorativeness score in [0, 1], combining:
        - higher natural material ratio -> more restorative,
        - mid-level visual entropy -> more restorative,
        - lower clutter and processing load -> more restorative,
      with clearly documented weights.
    - maps this score into a 3-level label {low, mid, high}.
    - appends:
        - a BN-like node to `bn.nodes` with `name="affect.restorative_h1"` and a one-hot
          posterior on the chosen bin, and
        - a derived tag to `tags` with `status="derived"`, `raw_value` equal to the numeric
          score, and `bin` equal to the label.
    - fails safely (returns no node/tag) if fewer than two of the required features are
      available for a given image.

- Governance:
  * `governance/TODO_running_list.md` updated with a v3.4.32 status entry describing this
    H1 restorativeness heuristic as a scaffold for future calibration and rule-family work.

All existing files from v3.4.31 are preserved; this version is strictly additive.
