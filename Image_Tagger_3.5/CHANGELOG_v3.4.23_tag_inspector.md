# Image Tagger v3.4.23 — Tag Inspector wiring

This version builds on v3.4.22_canon_guard_fixed and adds:

- A new supervisor endpoint: GET /v1/monitor/image/{image_id}/inspector
  which aggregates:
    * basic image metadata + /static URL,
    * science_pipeline_* Validation rows,
    * composite indices + bins from the BN export helpers,
    * a BN-style node summary for each candidate index,
    * per-user validations for the image.
- An upgraded Tag Inspector drawer in the Supervisor UI that:
    * fetches the inspector payload,
    * shows an image preview and science snapshot,
    * lists composite indices (BN inputs) with bins/values,
    * lists raw science attributes,
    * lists human validations with dwell time and timestamps.

All existing files from v3.4.22 are preserved; this version is strictly additive.
