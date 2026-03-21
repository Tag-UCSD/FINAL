# Image Tagger v3.4.31 — BN canon sanity tests

This version builds on v3.4.30_pipeline_health_tests and adds higher-level
"canon sanity" tests for the BN export and index catalog.

- Tests:
  * New `tests/test_bn_canon_sanity.py`:
    - `test_index_catalog_candidate_entries_are_well_formed` verifies that
      every candidate BN index in `backend/science/index_catalog.py` has:
        - a non-empty `label` and `description`,
        - a valid `type` in {"float", "int", "str"},
        - a `bins` spec with a non-empty `field` and list of `values`,
          and, when 3 values are present, that they are exactly {low, mid, high}.
    - `test_bn_export_respects_index_catalog_canon` seeds synthetic Validation
      rows for all candidate indices and their bin fields, calls
      `export_bn_snapshot`, and asserts:
        - `BNRow.indices.keys()` equals the candidate index key set and all
          values are non-None.
        - All expected bin fields appear in `BNRow.bins`, with labels from
          {low, mid, high} where configured.

- Governance:
  * `governance/TODO_running_list.md` updated with a v3.4.31 status entry
    describing the new BN canon sanity tests as part of the science/BN
    hardening track.

All existing files from v3.4.30 are preserved; this version is strictly additive.
