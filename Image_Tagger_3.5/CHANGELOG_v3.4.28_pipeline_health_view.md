# Image Tagger v3.4.28 — Science pipeline health view

This version builds on v3.4.27_priority_help and adds a dedicated pipeline health view for supervisors:

- Monitor app:
  * Added a "Science pipeline health" section beneath the top metrics grid.
  * The panel calls `/api/v1/debug/pipeline_health` (via a `debugApi` client) during the main `loadData` call
    and on demand via a "Re-check" button.
  * It displays:
    - Import status (OK/FAILED).
    - OpenCV availability.
    - Analyzers by tier with their `requires`/`provides` contracts.
    - Any warnings or analyzer instantiation errors returned by the endpoint.
  * Includes inline explanatory text clarifying that this is a contracts/instantiation check, not a full
    image-processing run, and that persistent failures should be escalated to engineering.

- Governance:
  * Updated `governance/TODO_running_list.md` with a v3.4.28 status entry marking the pipeline health
    view as implemented.

All existing files from v3.4.27 are preserved; this version is strictly additive.
