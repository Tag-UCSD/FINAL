# Image Tagger v3.4.30 — Pipeline health endpoint tests

This version builds on v3.4.29_tag_inspector_tests and adds a smoketest for the
`/api/v1/debug/pipeline_health` endpoint.

- Tests:
  * New `tests/test_pipeline_health_debug.py`:
    - Uses FastAPI's `TestClient` to hit `/api/v1/debug/pipeline_health`.
    - Asserts HTTP 200.
    - Validates that the response is a dict with:
      - `import_ok` (bool),
      - `cv2_available` (bool),
      - `analyzers_by_tier` (dict),
      - optional `warnings` and `analyzer_errors` (lists, if present).
    - Performs a light shape check over `analyzers_by_tier`, ensuring each analyzer entry
      includes `name`, `tier`, `requires`, and `provides`.

- Governance:
  * `governance/TODO_running_list.md` updated with a v3.4.30 status note marking
    TODO item 4.3 (pipeline health endpoint tests) as implemented.

All existing files from v3.4.29 are preserved; this version is strictly additive.
