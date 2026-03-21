# Image Tagger v3.4.62 — Science Tag Coverage Enforced

This release builds directly on v3.4.60 (semantic_tags_vlm base) and
includes two governance-focused changes:

## 1. Science Tag Coverage Map (Sprint A recap)

- New script: `scripts/generate_tag_coverage.py`.
- Produces a machine-readable snapshot at `science_tag_coverage_v1.json` that
  summarises:
    - total known feature keys
    - which keys have at least one compute implementation
    - which keys are marked as stubs
    - source_type classification:
      - `math_or_deterministic`
      - `vlm_cognitive`
      - `vlm_semantic`
      - `stub_only`
      - `unassigned` (should remain zero).
- Writes a human-readable summary to `docs/SCIENCE_TAG_MAP.md`.

## 2. Governance Enforcement for Tag Coverage (Sprint B)

- `v3_governance.yml` gains a new constraint flag:
    - `constraints.enforce_science_tag_coverage: true`
- `scripts/guardian.py` now includes `_check_science_tag_coverage(...)` and
  calls it from `verify(...)`.
- When the flag is enabled, `guardian verify` will fail if:
    - `science_tag_coverage_v1.json` is missing or unreadable; or
    - any feature keys are reported with `source_type == "unassigned"`.
- This ensures that every feature key in the union of registry/stub/computed
  keys is either:
    - wired via a math/VLM analyzer, or
    - explicitly tracked as a stub in `backend/science/feature_stubs.py`.

## Notes

- In this snapshot, the meta section of `science_tag_coverage_v1.json`
  reports zero `unassigned` keys; all tracked features are either wired or
  explicitly stubbed.
- Legacy discrepancies between `governance.lock` and the current working
  tree (e.g. missing __pycache__ files or changed hashes in protected
  modules) will still cause `guardian verify` to fail until the baseline is
  consciously refreshed via `guardian freeze` in the target environment.
