# Running TODO / Sprint Backlog (Governance-Tracked)

This file is the canonical, running TODO list for Image Tagger.
It is maintained across sprints and only **appended to**, never pruned.
Each item is written so that a future maintainer (human or AI) can implement it
without needing the original conversation for context.

Items are grouped by subsystem. When a task is completed in a later version,
it should be **struck through** and annotated with the version number
(e.g., ~~Done in v3.4.27~~).

---

## 1. Tag Inspector Stack (Supervisor / Monitor)

1.1 **Inspector endpoint tests**
- **Goal:** Protect `/api/v1/monitor/image/{image_id}/inspector` from regressions.
- **What to implement:**
  - Pytest-based tests that:
    - Seed an image with a few `science_pipeline_*` `Validation` rows and a couple of human validations.
    - Call the inspector endpoint and assert the presence and shape of:
      - `image`, `pipeline`, `features`, `tags`, `bn`, `validations`.
    - Check that feature keys and validation keys include known seeded values.
    - Assert 404 behavior for non-existent `image_id`.
- **Hints:** Use the same factories/fixtures used by BN export tests so canon and inspector evolve together.

1.2 **Decouple inspector from private BN helpers**
- **Goal:** Avoid hard-coding inspector against `_collect_*` private helpers in `v1_bn_export`.
- **What to implement:**
  - Create a small helper module (e.g., `backend/science/bn_helpers.py`) that:
    - Exposes public functions:
      - `collect_indices_for_image(db, image_id, index_keys) -> Dict[str, float]`
      - `collect_bins_for_image(db, image_id) -> Dict[str, str]`
      - `compute_irr_for_image(db, image_id) -> Optional[float]`
    - Internally reuses the existing BN export logic.
  - Refactor both BN export endpoint(s) and inspector endpoint to call these helpers.
- **Hints:** Keep helper signatures minimal; avoid leaking Response objects or HTTP concerns into science helpers.

1.3 **Pipeline status fields**
- **Goal:** Make `pipeline.overall_status` and `pipeline.analyzers_run` in the inspector payload informative.
- **What to implement:**
  - Define a per-image pipeline run marker (e.g., a `PipelineRun` table or a status field attached to `Image`).
  - Populate it whenever the science pipeline runs (batch or single).
  - In the inspector endpoint, set:
    - `overall_status` to something like `"not_run" | "ok" | "partial" | "error"`.
    - `analyzers_run` to a list of analyzer IDs or names that completed.
- **Hints:** Even a coarse first pass (e.g., `"ok"` vs `"not_run"`) is useful and can be refined later.

1.4 **Rule-level explanations (first family)**
- **Goal:** Start surfacing “why this psych tag or index fired” in the inspector payload.
- **What to implement:**
  - For at least one small family of indices (e.g., restorativeness and clutter), add:
    - A data structure describing which science attributes feed which index and via what thresholds/rules.
    - A helper that, given an image’s features, computes:
      - which rules fired,
      - their confidence, and
      - a short natural-language explanation string.
  - Extend the inspector payload with a `rules` block per index.
- **Hints:** Start small and explicit; this is about interpretability, not speed.

---

## 2. GUI Role Help & Onboarding

2.1 **Monitor dashboard help strip**
- **Goal:** Explain the Monitor dashboard as the “control tower” for supervisors.
- **What to implement:**
  - A small inline help strip (similar to Workbench and Explorer) that:
    - States that this view is for supervisors / PIs.
    - Explains what the key metrics mean (queue length, throughput, IRR, error counts).
    - Suggests when to drill into Tag Inspector vs escalate pipeline issues.
- **Hints:** Reuse the HelpCircle pattern and keep text under ~5 bullet points.

2.2 **Admin: provider / API configuration help**
- **Goal:** Avoid accidental misconfiguration of OpenAI/Anthropic/Gemini providers.
- **What to implement:**
  - Inline help near provider/API config that:
    - Clarifies that this is a sensitive, system-wide configuration surface.
    - Explains the recommended defaults and how to run a smoke test after changing keys/models.
    - Notes that VLM/materials paths may be disabled or degraded if configuration is incomplete.
- **Hints:** Include a short “After editing, run: [X]” note pointing to a concrete test (e.g., small VLM micro-loop).

2.3 **Admin: schema / index catalog help**
- **Goal:** Make it explicit that index/catalog edits affect BN inputs and inspector views.
- **What to implement:**
  - A help strip or tooltip cluster near index/catalog editing UI that:
    - States that index definitions determine which features are exported to BN and how Tag Inspector labels indices.
    - Encourages small, versioned changes (e.g., canon v1.0, v1.1).
    - Suggests review by at least one other scientist before changing canonical indices.
- **Hints:** Consider a “proposed vs active” index state to allow staged rollouts.

2.4 **BN export screen help**
- **Goal:** Help users understand what they are exporting and how to use it downstream.
- **What to implement:**
  - A help strip on the BN export / dataset export page that:
    - Explains that each row is an image/case.
    - Describes what columns correspond to (indices, features, tags, provenance).
    - Notes how to use the export for:
      - BN learning,
      - regression models,
      - cross-cultural comparisons.
- **Hints:** Point to a small example notebook path (if present) that demonstrates loading the export.

2.5 **Pipeline health / debug panel help**
- **Goal:** Explain how to interpret the `/v1/debug/pipeline_health` UI.
- **What to implement:**
  - Brief inline help describing:
    - What each tier means (e.g., L0 color, L1 texture, L2 fractal, etc.).
    - How to interpret pass/fail/partial indicators.
    - When a supervisor should escalate to an engineer.
- **Hints:** Use conservative language; this is a diagnostic panel, not a guarantee of scientific validity.

---

## 3. Scientific Validation & Norming

3.1 **Minimal validation harness**
- **Goal:** Establish a concrete path from computed features to human ratings/behavior.
- **What to implement:**
  - A small, well-documented script or module that:
    - Takes a curated set of images with human ratings for a few target constructs (e.g., restorativeness, stress, coziness).
    - Computes correlations or simple regressions between tiered features/indices and those ratings.
    - Writes out a report (e.g., CSV/Markdown) with effect sizes and directions.
- **Hints:** Start with a single scale (e.g., restoration) and one or two feature families to avoid overfitting.

3.2 **Population baselines by room type / culture**
- **Goal:** Support Goldilocks-style reasoning (mid-level optimality) and cultural modulation.
- **What to implement:**
  - A mechanism to:
    - Group images by room type (e.g., bedroom, office, café) and culture (e.g., India vs US).
    - Compute baseline distributions for key features (e.g., fractal metrics, clutter entropy, color saturation).
    - Store these baselines in a way that BN rules and Tag Inspector can reference.
- **Hints:** Treat baselines as versioned, with clear sample sizes, so future data can refine them without breaking existing analyses.

---

## 4. Governance & Integrity Extensions

4.1 **Syntax and AST guards**
- **Goal:** Ensure no syntactically broken Python modules ship in critical paths.
- **What to implement:**
  - A guard script (or extension of `program_integrity_guard.py`) that:
    - Walks all non-archive `.py` files in `backend/`, `scripts/`, and `tests/`.
    - Tries to `ast.parse` each file and reports failures.
  - CI and `install.sh` should fail if any AST parse fails.
- **Hints:** Respect existing exclude lists for archived/experimental code, but keep them minimal and documented.

4.2 **Inspector tests integrated into CI**
- **Goal:** Make the Tag Inspector a first-class citizen in governance.
- **What to implement:**
  - Ensure the inspector tests from Section 1.1 run under the main test suite.
  - Add a short note in `governance/README.md` (or equivalent) stating that Tag Inspector is covered by governance and must remain green for a GO release.
- **Hints:** Tag these tests with a marker (e.g., `@pytest.mark.governance`) if you need to separate quick vs full test runs.

---

## Status updates — v3.4.27_priority_help

The following high-priority GUI help items now have concrete implementations:

- **2.1 Monitor dashboard help strip** — Implemented in `frontend/apps/monitor/src/App.jsx` as an inline "How to use the Supervisor dashboard" box beneath the controls/status row.
- **2.2 Admin: provider / API configuration help** — Implemented in `frontend/apps/admin/src/App.jsx` inside `VLMConfigPanel` as a "VLM configuration: handle with care" help strip under the VLM Engine header.
- **2.4 BN export / training export screen help** — Implemented in `frontend/apps/admin/src/App.jsx` inside the Training Export card as a "What this export is for" help strip explaining downstream BN/regression/ML use.

Still pending (no dedicated UI surface yet or requires new panel):

- **2.3 Admin: schema / index catalog help** — Will be added once a schema/index management UI is exposed.
- **2.5 Pipeline health / debug panel help** — Will be added when the `/v1/debug/pipeline_health` results are surfaced in a dedicated debug/monitoring view.

---

## Status updates — v3.4.28_pipeline_health_view

- **2.5 Pipeline health / debug panel help** — Implemented as a "Science pipeline health" section in
  `frontend/apps/monitor/src/App.jsx`, backed by `/api/v1/debug/pipeline_health`. The panel:
    * Loads on initial dashboard fetch (via `loadData`).
    * Shows import status, OpenCV availability, analyzers by tier (with requires/provides),
      and any warnings/analyzer_errors.
    * Includes inline guidance explaining that this is a contracts/instantiation check rather than
      a full science run, and that supervisors should escalate persistent failures to engineering.

## 4.3 Pipeline health endpoint tests
- **Goal:** Ensure `/api/v1/debug/pipeline_health` has a stable, well-tested shape and fails loudly if its contract changes.
- **What to implement:**
  - Add a pytest module (e.g., `tests/test_pipeline_health_debug.py`) that:
    - Calls the endpoint via the FastAPI test client.
    - Asserts the presence and types of top-level fields: `import_ok` (bool), `cv2_available` (bool),
      `analyzers_by_tier` (mapping), and optional lists `warnings` and `analyzer_errors`.
    - Optionally validates that at least one analyzer is present in `analyzers_by_tier` in a seeded test environment.
  - Keep tests tolerant to minor content changes (e.g., different analyzer names) but strict about overall shape.
- **Hints:** Mirror the patterns used in `test_bn_export_smoke.py` and other existing tests that hit API endpoints.

## 4.4 Governance README note for pipeline health
- **Goal:** Make the pipeline health view part of the explicit governance surface for GO/NO-GO releases.
- **What to implement:**
  - Update `governance/README.md` (or create it if missing) to:
    - Describe the role of `/api/v1/debug/pipeline_health` and the Monitor's "Science pipeline health" panel.
    - State that a GO release requires this endpoint and panel to be functioning (no errors, sensible content).
    - Reference the associated tests (Section 4.3) and indicate that they must pass in CI.
- **Hints:** Keep this note short and operational, framed as an acceptance criterion that PIs/supervisors can understand.

---

## Status updates — v3.4.30_pipeline_health_tests

- **4.3 Pipeline health endpoint tests** — Implemented as `tests/test_pipeline_health_debug.py`,
  which calls `/api/v1/debug/pipeline_health` via FastAPI's TestClient and asserts:
    * Response status code is 200.
    * Top-level fields `import_ok`, `cv2_available`, and `analyzers_by_tier` are present and of the
      expected types (bool, bool, dict).
    * Optional fields `warnings` and `analyzer_errors` (if present) are lists.
    * Each analyzer entry (if present) includes `name`, `tier`, `requires`, and `provides` keys.

- **4.4 Governance README note for pipeline health** — Still pending; will be implemented as a short
  acceptance-criteria section in `governance/README.md` that ties the Monitor pipeline health panel and
  `/api/v1/debug/pipeline_health` tests to GO/NO-GO decisions.

---

## Status updates — v3.4.31_bn_canon_sanity

- Added `tests/test_bn_canon_sanity.py` to exercise the BN-facing canon:
  * `test_index_catalog_candidate_entries_are_well_formed` checks that each
    candidate BN index in `index_catalog` has a label, description, type, and
    bins spec with non-empty `field` and `values` (and that 3-level bins use
    `{low, mid, high}`).
  * `test_bn_export_respects_index_catalog_canon` seeds synthetic Validation rows
    for all candidate indices and their bin fields, calls `export_bn_snapshot`,
    and asserts:
      - `BNRow.indices.keys()` == the candidate index key set and all values are non-None.
      - All expected bin fields appear in `BNRow.bins`, with labels drawn from
        `{low, mid, high}` where applicable.

---

## Status updates — v3.4.32_restorativeness_H1

- Added `_build_restorativeness_heuristic_node` to `backend/api/v1_supervision.py` and wired it
  into the Tag Inspector (`/v1/monitor/image/{image_id}/inspector`):
    * The helper reads CNfA fluency + biophilia features from `features` (notably
      `cnfa.biophilic.natural_material_ratio`, `cnfa.fluency.visual_entropy_spatial`,
      `cnfa.fluency.clutter_density_count`, and `cnfa.fluency.processing_load_proxy`).
    * It computes a simple, explicitly *heuristic* restorativeness score in [0, 1] and maps
      it into a coarse 3-bin label {low, mid, high}.
    * It then appends:
        - a BN-like node dict to `bn.nodes` with `name="affect.restorative_h1"` and a
          one-hot posterior over the chosen bin; and
        - a derived tag to `tags` with `status="derived"`, `key="affect.restorative_h1"`,
          `raw_value` equal to the numeric score, and `bin` equal to the 3-level label.
    * If fewer than two of the required features are available, the helper returns no node/tag,
      so the heuristic is fail-safe and purely additive.
- This is the first explicit "restorativeness" rule family (H1) in Tag Inspector and is
  intended as a scaffold for later calibration against human rating datasets and cultural
  baselines.

---

## Status updates — v3.4.33_admin_killswitch_vlm_depth_fix

- **Admin kill switch build break (Gemini ruthless panel)** — Fixed in `frontend/apps/admin/src/App.jsx` by rewriting `handleKillSwitch` so it:
  - Calls `/api/v1/admin/kill-switch?active=<bool>` with a clear busy/error state.
  - Updates both `budget` and `killSwitchActive` from the returned `BudgetStatus`.
  - Handles errors via `console.error` and a guarded `setError(...)`.
  - Always clears `killBusy` in a `finally` block.
- **Admin bulk upload endpoint (`/upload`)** — Restored the clean async implementation in `backend/api/v1_admin.py` from the archived v3.4.13 snapshot and adapted it so:
  - The DB `Image.storage_path` stores the relative `unique_name`.
  - `storage_paths` returned to the Admin UI still use full filesystem paths for operator feedback.
  - The endpoint is `async` with `await f.read()` inside the function body (removing the prior `await`-outside-async SyntaxError).
- **Depth analyzer scientific clarification** — Updated `backend/science/spatial/depth.py`:
  - Module docstring now clearly states this module provides 2D edge/clutter heuristics as *proxies* for depth-like qualities (openness, refuge, isovist area).
  - `DepthAnalyzer` class docstring now marks the outputs as heuristic indicators, pending integration of true monocular depth maps.
- **VLM JSON robustness** — Introduced `_safe_json_loads` in `backend/services/vlm.py` and wired it into:
  - `OpenAIEngine.analyze_image`
  - `GeminiEngine.analyze_image`
  The helper:
  - Strips Markdown fences (```json / ```).
  - Attempts `json.loads` on the cleaned string.
  - On failure, extracts the first `{...}` span before re-raising a `JSONDecodeError`.
  This covers the most common “JSON wrapped in commentary/markdown” cases without doing unsafe free-form repair.
- **Python syntax health (active code)** — All active (non-`archive/`) `.py` files pass an AST/syntax check after these changes. Archive trees remain untouched as historical snapshots.

---

## Status updates — v3.4.34_G1_admin_killswitch_vlm_tests

- **Admin kill-switch tests** — Added `tests/test_admin_killswitch.py` to:
  - Seed a paid `ToolConfig` if none exist.
  - Exercise `/api/v1/admin/budget` and assert it returns a boolean `is_kill_switched`.
  - Call `/api/v1/admin/kill-switch?active=true` as an admin and assert:
    - The response marks `is_kill_switched == True`.
    - No paid `ToolConfig` entries remain enabled in the database.
- **VLM JSON robustness tests** — Added `tests/test_vlm_safe_json_loads.py` to unit-test `_safe_json_loads`:
  - Plain JSON object.
  - ```json fenced blocks.
  - Generic ``` fenced blocks.
  - JSON with leading/trailing commentary.
  - Ensures a real `json.JSONDecodeError` is raised on non-JSON garbage.
- These tests codify the Gemini ruthless panel’s concerns about the Admin kill-switch build path and VLM JSON parsing, turning those into stable, regresssion-resistant contracts.


---

## Status updates — v3.4.35_science_bn_restH1_closure

- **BN canon + export sanities** — Confirmed that:
  - `tests/test_bn_canon_sanity.py` enforces that all `candidate_bn_input` indices in
    `backend/science/index_catalog.py` have labels, descriptions, typed bins, and that
    `export_bn_snapshot` exposes exactly those indices and bin fields for a seeded image.
  - `tests/test_bn_export_smoke.py` exercises a full BN snapshot round-trip, asserting
    that a synthetic continuous index value and its corresponding bin field map to the
    expected `low`/`mid`/`high` label.
- **Restorativeness H1 integration** — The H1 restorativeness node (`affect.restorative_h1`)
  is now part of the Tag Inspector BN payload, and
  `tests/test_monitor_tag_inspector.py::test_restorativeness_h1_appears_in_tag_inspector`
  covers the end-to-end path from synthetic science-like `Validation` rows to:
  - a BN node with posterior `{"high": 1.0}`, and
  - a derived Tag Inspector tag with `bin == "high"`.
- Together with the governance and syntax guards, this closes the initial science/BN sprint:
  BN inputs are canon-checked and restorativeness H1 is a stable, regression-tested contract.
